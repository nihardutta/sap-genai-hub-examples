"""
orchestration-service.py
==============================================================================================================
Orchestration Service Tutorial
----------------------------------------------
About this project
This is NOT a replacement for SAP's Generative AI Hub or its Orchestration Service. It is an educational
re-creation of SAP's orchestration tutorial, for people who do not have access to SAP AI Core and its
services. It uses LangChain, Microsoft Presidio and freely available LLMs to explore the same concepts.

Original SAP notebook (the reference for this example):
https://github.com/SAP-samples/ai-core-samples/blob/main/10_Learning_Journeys/Orchestration/orchestration-service.ipynb

The SAP notebook shows how to use the SDK with the Orchestration Service to build AI workflows from modules
such as templating, large language models (LLMs), data masking and content filtering. This file shows how
the same ideas can be put together with open-source tools.

The flow follows the SAP tutorial, using LangChain in place of the SAP SDK:
    1. Define a template of messages with placeholders and default values.
    2. Define the LLM.
    3. Combine them into a chain (the LangChain counterpart of SAP's OrchestrationConfig).
    4. Run the chain with input values (the counterpart of SAP's OrchestrationService.run).
    5. Try the optional modules: data masking (Microsoft Presidio) and content filtering.

Author: Nihar Dutta
Repository: https://github.com/nihardutta/sap-genai-hub-examples.git
Disclaimer: For education purpose only. Not for productive use. As usual, check for vulnerabilities of any open-source 
codes before you hit run.
===============================================================================================================
Install these packages before running this module.
    uv add langchain langchain-core python-dotenv langchain-openai presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg
"""

# Langchain and other imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from helper.llm_factory import get_llm
from dotenv import load_dotenv

# Helper modules
from helper.message_template import MessageTemplate
from helper.presidio_masking import PresidioMasker, build_analyzer
from helper.content_filtering import (
    ContentFilter,
    ContentFilterError,
    SafetyAssessment,
    Threshold,
    load_content,
)

load_dotenv()

# Pass the provider name explicitly
MODEL_PROVIDER = 'OpenRouter'

# Hard-coding known ORGANIZATION values
KNOWN_ORGANISATIONS = ["Contoso Pty Ltd", "Fabrikam", "Northwind Logistics"]

def contentFilter():
    # Initialise llm with temperature 0
    llm = get_llm(MODEL_PROVIDER, 0.0)
    safety_classifier = llm.with_structured_output(SafetyAssessment)

    # Same thresholds as the SAP notebook
    input_filter = ContentFilter(
        safety_classifier,
        hate=Threshold.ALLOW_SAFE,
        violence=Threshold.ALLOW_SAFE,
        self_harm=Threshold.ALLOW_SAFE,
        sexual=Threshold.ALLOW_SAFE,
    )
    output_filter = ContentFilter(
        safety_classifier,
        hate=Threshold.ALLOW_SAFE,
        violence=Threshold.ALLOW_SAFE_LOW,
        self_harm=Threshold.ALLOW_SAFE_LOW_MEDIUM,
        sexual=Threshold.ALLOW_ALL,
    )

    filter_template = MessageTemplate(
        messages=[
            SystemMessage("You are a helpful AI assistant."),
            HumanMessage("{{?text}}"),
        ]
    )

    filtered_chain = (
        input_filter.as_input_filter()
        | filter_template.as_runnable()
        | llm
        | output_filter.as_output_filter()
        | StrOutputParser()
    )

    # Invoking content filtering chain with different type of messages
    # msg = load_content("msg_clean")     # No error
    msg = load_content("msg_h")         # Content blocked by input filter: hate severity 2 (allowed up to 0)
    # msg = load_content("msg_v")         # Content blocked by input filter: violence severity 2 (allowed up to 0)
    # msg = load_content("msg_sh")        # Content blocked by input filter: self_harm severity 2 (allowed up to 0)          
    # msg = load_content("msg_s")         # Content blocked by input filter: sexual severity 2 (allowed up to 0)
    try:
        result = filtered_chain.invoke({"text": msg})
        print(result)
    except ContentFilterError as error:
        print(error.message)


def translate():
    template = MessageTemplate(
        messages=[
            SystemMessage("You are a helpful translation assistant."),
            HumanMessage("Translate the following text to {{?to_lang}}: {{?text}}"),
        ],
        defaults={"to_lang": "German"},
    )

    # Preview the messages that will be sent to the model
    # for message in template.format_messages({"text": "The Orchestration Service is working!"}):
    #     message.pretty_print()

    # Initialise llm
    llm = get_llm(MODEL_PROVIDER)

    # Build and run the chain
    chain = template.as_runnable() | llm | StrOutputParser()
    # result = chain.invoke({"text": "The Orchestration Service is working!"})
    # print(result)

    # Override the default language, just like passing another TemplateValue in SAP
    # print(chain.invoke({"text": "The Orchestration Service is working!", "to_lang": "French"}))

    # Raw result similar to SAP's response object
    raw_chain = template.as_runnable() | llm
    response = raw_chain.invoke({"text": "Science may never come up with a better office communication system than the coffee break.", "to_lang": "Bengali"})

    print("Type:  ", type(response).__name__)
    print("Text:  ", response.text)
    print("Usage: ", response.usage_metadata)

def presidioMasking():
    # Build the Presidio analyzer
    analyzer = build_analyzer()
    # print("Entities this analyzer can find:")
    # print(", ".join(sorted(analyzer.get_supported_entities())))

    # Select a CV file here
    with open("data/liam_cv.txt", "r", encoding="utf-8") as file:
        cv_as_string = file.read()

    ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "ORGANIZATION", "LOCATION"]

    # Anonymization missing some of the ORGANIZATION values
    # masker = PresidioMasker(analyzer, entities=ENTITIES, method="anonymization")
    # masker.report(cv_as_string)

    # Initialise llm
    llm = get_llm(MODEL_PROVIDER)

    # anonymization masking
    print("\n\n==================anonymization masking==================")
    masker = PresidioMasker(
        analyzer,
        entities=ENTITIES,
        method="anonymization",  # or "pseudonymization"
        deny_lists={"ORGANIZATION": KNOWN_ORGANISATIONS},
    )
    # masker.report(cv_as_string)
    # print()
    # print(masker.mask_text(cv_as_string))

    cv_template = MessageTemplate(
        messages=[
            SystemMessage("You are a helpful AI assistant."),
            HumanMessage("Summarize the following CV in 10 sentences: {{?orgCV}}"),
        ]
    )

    masked_chain = masker.as_runnable() | cv_template.as_runnable() | llm | StrOutputParser()

    result = masked_chain.invoke({"orgCV": cv_as_string})
    print(result)

    print("\n\n==================pseudonymization masking==================")
    # pseudonymization masking
    pseudo_masker = PresidioMasker(
        analyzer,
        entities=ENTITIES,
        method="pseudonymization",
        deny_lists={"ORGANIZATION": KNOWN_ORGANISATIONS},
    )

    print(pseudo_masker.mask_text(cv_as_string))
    print()

    pseudo_chain = (
        pseudo_masker.as_runnable()
        | cv_template.as_runnable()
        | llm
        | StrOutputParser()
        | pseudo_masker.unmask_runnable()
    )

    print(pseudo_chain.invoke({"orgCV": cv_as_string}))
    print("\nPlaceholders used:")
    for placeholder, value in pseudo_masker.mapping.items():
        print(f"  {placeholder:<24} {value!r}")


    
def main():
    # Translate using Langchain chain
    # translate()

    # Data Masking
    # presidioMasking()

    # Content Filtering
    contentFilter()
    
    # print("Nothing to be called")


if __name__ == "__main__":
    main()
