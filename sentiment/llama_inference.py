def build_prompt(sentence: str, entity: str) -> str:
    return (
        f"Analizează următoarea propoziție și clasifică atitudinea față de entitatea „{entity}”.\n"
        f"Răspunsul trebuie să fie un singur cuvânt, exact unul dintre: pozitiv, negativ sau neutru.\n\n"
        f"Propoziție: {sentence}\n"
        f"Răspuns:"
    )


