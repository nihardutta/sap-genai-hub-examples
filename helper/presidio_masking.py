"""
presidio_masking.py
Helpers to mask personal information with Microsoft Presidio.
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_core.runnables import Runnable, RunnableLambda

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)  # hide start up warnings

SPACY_MODEL = "en_core_web_lg"  # must match the model downloaded
PHONE_REGIONS = ("AU", "US", "UK", "DE", "IN", "CA")


# Class to mask personal information with Microsoft Presidio.
class PresidioMasker:
    """Masks personal information in the chain input with Microsoft Presidio."""

    def __init__(
        self,
        analyzer: AnalyzerEngine,
        entities: List[str],
        method: str = "anonymization",
        deny_lists: Optional[Dict[str, List[str]]] = None,
        score_threshold: float = 0.35,
        fields: Optional[List[str]] = None,
        language: str = "en",
    ):
        """
        analyzer:        a Presidio AnalyzerEngine
        entities:        Presidio entity names to mask, e.g. ["PERSON", "EMAIL_ADDRESS"]
        method:          "anonymization" or "pseudonymization"
        deny_lists:      exact terms to always mask, by entity, e.g. {"ORGANIZATION": ["Contoso Pty Ltd"]}
        score_threshold: minimum Presidio confidence (0 to 1) for a finding to be masked
        fields:          chain input keys to mask; None masks every text input
        """
        if method not in ("anonymization", "pseudonymization"):
            raise ValueError("method must be 'anonymization' or 'pseudonymization'")
        self.analyzer = analyzer
        self.anonymizer = AnonymizerEngine()
        self.method = method
        self.score_threshold = score_threshold
        self.fields = fields
        self.language = language
        self.deny_lists = dict(deny_lists or {})
        self.entities = list(dict.fromkeys([*entities, *self.deny_lists]))
        self.ad_hoc_recognizers = [
            PatternRecognizer(
                supported_entity=entity,
                deny_list=terms,
                name=f"deny_list_{entity.lower()}",
            )
            for entity, terms in self.deny_lists.items()
        ]
        self.mapping: Dict[str, str] = {}  # placeholder -> original value (pseudonymisation only)

    def analyze(self, text: str):
        return self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
            ad_hoc_recognizers=self.ad_hoc_recognizers or None,
        )

    def report(self, text: str) -> None:
        """Print what Presidio found, so you can check it before sending anything."""
        results = sorted(self.analyze(text), key=lambda r: r.start)
        print(f"{'Entity':<14} {'Score':>5}  {'Found by':<22} Text")
        for r in results:
            found = text[r.start:r.end].replace("\n", "\\n")
            recognizer = r.recognition_metadata.get("recognizer_name", "")
            print(f"{r.entity_type:<14} {r.score:>5.2f}  {recognizer:<22} {found}")

    def _placeholder(self, entity: str, original: str) -> str:
        for placeholder, value in self.mapping.items():
            if value == original and placeholder.startswith(f"MASKED_{entity}_"):
                return placeholder
        number = 1 + sum(1 for p in self.mapping if p.startswith(f"MASKED_{entity}_"))
        placeholder = f"MASKED_{entity}_{number}"
        self.mapping[placeholder] = original
        return placeholder

    def mask_text(self, text: str) -> str:
        results = self.analyze(text)
        if self.method == "anonymization":
            operators = {
                entity: OperatorConfig("replace", {"new_value": f"MASKED_{entity}"})
                for entity in self.entities
            }
        else:
            # Number the placeholders in reading order (Presidio replaces from the end backwards)
            for r in sorted(results, key=lambda r: r.start):
                self._placeholder(r.entity_type, text[r.start:r.end])
            operators = {
                entity: OperatorConfig(
                    "custom", {"lambda": lambda value, entity=entity: self._placeholder(entity, value)}
                )
                for entity in self.entities
            }
        return self.anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text

    def mask_inputs(self, values: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: self.mask_text(value)
            if isinstance(value, str) and (self.fields is None or key in self.fields)
            else value
            for key, value in values.items()
        }

    def unmask(self, text: str) -> str:
        """Put the original values back (pseudonymisation only)."""
        # Longest placeholders first, so MASKED_PERSON_10 is not caught by MASKED_PERSON_1
        for placeholder in sorted(self.mapping, key=len, reverse=True):
            text = text.replace(placeholder, self.mapping[placeholder])
        return text

    def reset(self) -> None:
        self.mapping.clear()

    def as_runnable(self) -> Runnable:
        return RunnableLambda(self.mask_inputs, name="PresidioMasker")

    def unmask_runnable(self) -> Runnable:
        return RunnableLambda(self.unmask, name="PresidioUnmasker")


def build_analyzer(spacy_model: str = SPACY_MODEL, phone_regions=PHONE_REGIONS) -> AnalyzerEngine:
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
        "ner_model_configuration": {
            # spaCy label -> Presidio entity
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "LOC": "LOCATION",
                "GPE": "LOCATION",
                "FAC": "LOCATION",
                "ORG": "ORGANIZATION",  # ignored by Presidio's defaults, switched on here
                "NORP": "NRP",
            },
            "labels_to_ignore": [
                "CARDINAL", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY",
                "ORDINAL", "LANGUAGE", "LAW", "EVENT", "PRODUCT", "WORK_OF_ART",
            ],
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": [],
        },
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

    registry = RecognizerRegistry(supported_languages=["en"])
    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
    registry.remove_recognizer("PhoneRecognizer")
    registry.add_recognizer(PhoneRecognizer(supported_regions=phone_regions))

    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])
