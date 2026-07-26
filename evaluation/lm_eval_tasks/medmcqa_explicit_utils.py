"""Prompt conversion for the explicitly namespaced MedMCQA dataset."""


def doc_to_text(doc) -> str:
    options = {
        "A": doc["opa"],
        "B": doc["opb"],
        "C": doc["opc"],
        "D": doc["opd"],
    }
    choices = "\n".join(f"{label}. {text}" for label, text in options.items())
    return f"Question: {doc['question']}\nChoices:\n{choices}\nAnswer:"
