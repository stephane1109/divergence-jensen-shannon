# -*- coding: utf-8 -*-

import re
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


texte_a = """
Je me réveille souvent avec une sensation d’angoisse. J’ai l’impression que quelque chose va mal se passer. Je pense immédiatement à tout ce que je dois faire. Je sens une tension dans ma poitrine dès le matin. J’ai parfois du mal à respirer calmement. Je réfléchis beaucoup avant de prendre une décision. J’ai peur de me tromper. Je vérifie souvent plusieurs fois les mêmes choses. Je me sens épuisé mentalement. Je pense constamment au travail. Je crains de ne pas être à la hauteur. Je me mets facilement la pression. Je ressasse souvent des conversations anciennes. Je me demande si les autres me jugent. J’évite parfois de répondre au téléphone. Je préfère rester seul quand l’anxiété augmente. Je dors mal depuis plusieurs mois. Je me réveille souvent pendant la nuit. Je pense immédiatement aux problèmes du lendemain. J’ai du mal à arrêter mes pensées. Je sens mon corps tendu presque toute la journée. J’ai parfois l’impression de perdre le contrôle. Je me sens fragile émotionnellement. Je me fatigue rapidement dans les situations sociales. Je redoute les imprévus. Je supporte mal l’incertitude. Je voudrais retrouver un peu de calme. Je voudrais comprendre pourquoi cette anxiété revient sans arrêt. Je fais beaucoup d’efforts pour paraître normal devant les autres. Je me sens souvent seul avec mes pensées.
"""

texte_b = """
Je me sens anxieux presque tous les jours. Je pense constamment au travail. Je dors mal la nuit. J’évite les autres quand je suis stressé. Je voudrais réussir à me calmer.
"""


STOPWORDS_FR_SUPPLEMENTAIRES = {
    "c", "j", "l", "m", "n", "qu", "s", "t", "y"
}

MOTIF_MOTS = re.compile(
    r"[a-zàâçéèêëîïôûùüÿñæœ]+"
)


def charger_stopwords_francais():

    return set(
        stopwords.words("french")
    ) | STOPWORDS_FR_SUPPLEMENTAIRES


STOPWORDS_FR = charger_stopwords_francais()


def nettoyer_texte(texte):

    texte = (
        texte.lower()
        .replace("’", "'")
    )

    texte = re.sub(
        r"[']",
        " ",
        texte
    )

    mots = MOTIF_MOTS.findall(
        texte
    )

    mots_filtres = [
        mot for mot in mots
        if mot not in STOPWORDS_FR and len(mot) > 2
    ]

    return mots_filtres


def compter_mots(texte):

    return Counter(nettoyer_texte(texte))


def creer_vocabulaire(compteur_a, compteur_b):

    return sorted(
        set(compteur_a.keys()) |
        set(compteur_b.keys())
    )


def creer_distribution(compteur, vocabulaire):

    total = sum(compteur.values())

    if total == 0:
        raise ValueError(
            "Aucun mot n'a ete conserve apres le filtrage. "
            "Assouplissez les stopwords ou le pretraitement."
        )

    return np.array(
        [
            compteur.get(mot, 0) / total
            for mot in vocabulaire
        ],
        dtype=float
    )


def x_log2_x(x):

    if x <= 0:
        return 0.0

    return x * math.log2(x)


def calculer_contributions_jsd(p, q):

    m = 0.5 * (p + q)

    contributions = []

    for pi, qi, mi in zip(p, q, m):

        valeur = (
            -x_log2_x(mi)
            + 0.5 * x_log2_x(pi)
            + 0.5 * x_log2_x(qi)
        )

        contributions.append(valeur)

    return np.array(contributions)


def orienter_valeurs(p, q, contributions):

    valeurs = []

    for pi, qi, contribution in zip(p, q, contributions):

        if pi >= qi:
            valeurs.append(-contribution)

        else:
            valeurs.append(contribution)

    return np.array(valeurs)


def determiner_dominant(p_i, q_i):

    if p_i > q_i:
        return "Texte A"

    if q_i > p_i:
        return "Texte B"

    return "Égal"


def afficher_textes():

    print("\n" + "=" * 100)
    print("TEXTE A")
    print("=" * 100)

    print(texte_a.strip())

    print("\n" + "=" * 100)
    print("TEXTE B")
    print("=" * 100)

    print(texte_b.strip())


def afficher_tableau(
    titre,
    vocabulaire,
    p,
    q,
    contributions,
    top_n=25
):

    valeurs = orienter_valeurs(
        p,
        q,
        contributions
    )

    score_global = contributions.sum()

    indices = np.argsort(
        np.abs(valeurs)
    )[::-1][:top_n]

    print("\n" + "=" * 100)
    print(titre)
    print("=" * 100)

    print(
        f"Score global : "
        f"{contributions.sum():.6f}"
    )

    print(
        f"{'rang':>4s} "
        f"{'mot':20s} "
        f"{'score':>12s} "
        f"{'%JSD':>8s} "
        f"{'pA':>10s} "
        f"{'pB':>10s} "
        f"{'dominant':>12s}"
    )

    print("-" * 100)

    for rang, indice in enumerate(indices, start=1):

        mot = vocabulaire[indice]

        pourcentage_jsd = 0.0

        if score_global > 0:
            pourcentage_jsd = (
                contributions[indice] / score_global
            ) * 100

        dominant = determiner_dominant(
            p[indice],
            q[indice]
        )

        print(
            f"{rang:4d} "
            f"{mot:20s} "
            f"{contributions[indice]:12.6f} "
            f"{pourcentage_jsd:8.2f} "
            f"{p[indice]:10.4f} "
            f"{q[indice]:10.4f} "
            f"{dominant:>12s}"
        )


def tracer_sous_graphe(
    ax,
    titre,
    vocabulaire,
    p,
    q,
    contributions,
    indices=None,
    limite_x=None,
    top_n=25
):

    valeurs = orienter_valeurs(
        p,
        q,
        contributions
    )

    if indices is None:
        indices = np.argsort(
            np.abs(valeurs)
        )[::-1][:top_n]

    mots = [
        vocabulaire[i]
        for i in indices
    ][::-1]

    valeurs_selectionnees = [
        valeurs[i]
        for i in indices
    ][::-1]

    couleurs = [
        "#8aa84f"
        if valeur < 0
        else "#4f5f3a"
        for valeur in valeurs_selectionnees
    ]

    ax.barh(
        mots,
        valeurs_selectionnees,
        color=couleurs
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.set_title(
        titre,
        fontsize=10
    )

    ax.grid(
        axis="x",
        alpha=0.3
    )

    ax.tick_params(
        axis="y",
        labelsize=8
    )

    ax.tick_params(
        axis="x",
        labelsize=8
    )

    if limite_x is not None:
        ax.set_xlim(
            -limite_x,
            limite_x
        )


def exporter_resultats_txt(
    texte_a,
    texte_b,
    vocabulaire,
    p,
    q,
    contributions,
    taille_a,
    taille_b,
    chemin="resultats_jsd.txt",
    top_n=25
):

    with open(
        chemin,
        "w",
        encoding="utf-8"
    ) as fichier:

        fichier.write(
            "Analyse JSD\n\n"
        )

        fichier.write(
            "TEXTE A\n"
        )

        fichier.write(
            "=" * 100 + "\n"
        )

        fichier.write(
            texte_a.strip() + "\n\n"
        )

        fichier.write(
            f"Nombre de mots conservés Texte A : {taille_a}\n"
        )

        fichier.write(
            "\nTEXTE B\n"
        )

        fichier.write(
            "=" * 100 + "\n"
        )

        fichier.write(
            texte_b.strip() + "\n\n"
        )

        fichier.write(
            f"Nombre de mots conservés Texte B : {taille_b}\n"
        )

        fichier.write(
            f"\nVocabulaire total après filtrage : {len(vocabulaire)}\n"
        )

        fichier.write(
            f"Score global Divergence de Jensen Shannon : "
            f"{contributions.sum():.6f}\n\n"
        )

        valeurs = orienter_valeurs(
            p,
            q,
            contributions
        )

        score_global = contributions.sum()

        indices = np.argsort(
            np.abs(valeurs)
        )[::-1][:top_n]

        fichier.write(
            "=" * 100 + "\n"
        )

        fichier.write(
            "Divergence de Jensen Shannon\n"
        )

        fichier.write(
            "=" * 100 + "\n"
        )

        fichier.write(
            f"Score global : "
            f"{contributions.sum():.6f}\n"
        )

        fichier.write(
            f"{'rang':>4s} "
            f"{'mot':20s} "
            f"{'score':>12s} "
            f"{'%JSD':>8s} "
            f"{'pA':>10s} "
            f"{'pB':>10s} "
            f"{'dominant':>12s}\n"
        )

        for rang, indice in enumerate(indices, start=1):

            pourcentage_jsd = 0.0

            if score_global > 0:
                pourcentage_jsd = (
                    contributions[indice] / score_global
                ) * 100

            dominant = determiner_dominant(
                p[indice],
                q[indice]
            )

            fichier.write(
                f"{rang:4d} "
                f"{vocabulaire[indice]:20s} "
                f"{contributions[indice]:12.6f} "
                f"{pourcentage_jsd:8.2f} "
                f"{p[indice]:10.4f} "
                f"{q[indice]:10.4f} "
                f"{dominant:>12s}\n"
            )

        fichier.write("\n")


def analyser_et_tracer(
    texte_a,
    texte_b
):

    compteur_a = compter_mots(
        texte_a
    )

    compteur_b = compter_mots(
        texte_b
    )

    vocabulaire = creer_vocabulaire(
        compteur_a,
        compteur_b
    )

    p = creer_distribution(
        compteur_a,
        vocabulaire
    )

    q = creer_distribution(
        compteur_b,
        vocabulaire
    )

    taille_a = sum(
        compteur_a.values()
    )

    taille_b = sum(
        compteur_b.values()
    )

    contributions = calculer_contributions_jsd(
        p,
        q
    )

    score_jsd = contributions.sum()

    afficher_textes()

    print("\n" + "=" * 100)
    print("INFORMATIONS GÉNÉRALES")
    print("=" * 100)

    print(
        f"Nombre de mots conservés Texte A : "
        f"{taille_a}"
    )

    print(
        f"Nombre de mots conservés Texte B : "
        f"{taille_b}"
    )

    print(
        f"Vocabulaire total après filtrage : "
        f"{len(vocabulaire)}"
    )

    print(
        f"Score global Divergence de Jensen Shannon : "
        f"{score_jsd:.6f}"
    )

    afficher_tableau(
        "Divergence de Jensen Shannon",
        vocabulaire,
        p,
        q,
        contributions
    )

    exporter_resultats_txt(
        texte_a,
        texte_b,
        vocabulaire,
        p,
        q,
        contributions,
        taille_a,
        taille_b
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    valeurs = orienter_valeurs(
        p,
        q,
        contributions
    )

    limite_x = np.max(
        np.abs(valeurs)
    ) * 1.1

    tracer_sous_graphe(
        ax,
        "Divergence de Jensen Shannon",
        vocabulaire,
        p,
        q,
        contributions,
        limite_x=limite_x
    )

    ax.set_xlabel(
        "Gauche = Texte A | Droite = Texte B",
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        "jsd_texte_a_vs_b.png",
        dpi=300
    )

    plt.show()

    print(
        "\nGraphique exporté : "
        "jsd_texte_a_vs_b.png"
    )

    print(
        "Résultats exportés : "
        "resultats_jsd.txt"
    )


if __name__ == "__main__":

    analyser_et_tracer(
        texte_a,
        texte_b
    )
