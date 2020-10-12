import csv
import itertools
import sys
from typing import Dict, Literal, Set

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people: Dict[str, Dict],
                      one_gene: Set[str], two_genes: Set[str],
                      have_trait: Set[str]) -> float:
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    prod = 1.0
    for person, person_data in people.items():
        num_genes = get_num_genes(person, one_gene, two_genes)
        has_trait = person in have_trait
        father = person_data["father"]
        mother = person_data["mother"]

        # No parents listed. Calculate unconditional probability
        if father is None:
            assert mother is None
            # Get gene probability from `PROBS`
            gene_prob: float = PROBS["gene"][num_genes]

        # Parents listed. Calculate conditional probability
        else:
            assert mother is not None
            father_num_genes = get_num_genes(father, one_gene, two_genes)
            mother_num_genes = get_num_genes(mother, one_gene, two_genes)
            # 2 genes for child. Both parents have to give a bad gene
            if num_genes == 2:
                gene_prob = (get_inherit_gene_prob(father_num_genes, True)
                             * get_inherit_gene_prob(mother_num_genes, True))
            # 1 gene for child. One parent has to give a good gene, while the
            #   other has to give a bad gene
            elif num_genes == 1:
                gene_prob = (get_inherit_gene_prob(father_num_genes, True)
                             * get_inherit_gene_prob(mother_num_genes, False)
                             + get_inherit_gene_prob(father_num_genes, False)
                             * get_inherit_gene_prob(mother_num_genes, True))
            # 0 genes for child. Both parents have to give a good gene
            else:
                gene_prob = (get_inherit_gene_prob(father_num_genes, False)
                             * get_inherit_gene_prob(mother_num_genes, False))

        # Get trait probability from `PROBS`
        trait_prob_given_gene: float = PROBS["trait"][num_genes][has_trait]
        # P(gene ^ trait) = P(trait | gene) × P(gene)
        gene_and_trait_prob = trait_prob_given_gene * gene_prob

        prod *= gene_and_trait_prob

    return prod


def get_num_genes(person: str, one_gene: Set[str], two_genes: Set[str]) -> Literal[0, 1, 2]:
    """
    Returns the number of bad genes a person has in the scenario

    Args:
        person    {str}     : The name of the person
        one_gene  {set[str]}: Set of people with one bad gene
        two_genes {set[str]}: Set of people with two bad genes

    Returns:
        {Literal[0, 1, 2]}: The number of bad genes the person has
    """
    if person in two_genes:
        return 2
    elif person in one_gene:
        return 1
    else:
        return 0


def get_inherit_gene_prob(parent_num_genes: Literal[0, 1, 2], bad_gene: bool) -> float:
    """
    Returns the probability of passing on a good/bad gene

    Args:
        parent_num_genes {Literal[0, 1, 2]}: Number of bad genes the parent has
        bad_gene {bool}: `True` if we're calculating the possibility of
                           passing on a bad gene. `False` if we're calculating
                           that of a good gene

    Returns:
        {float}: The probability of given conditions
    """
    mutation_prob: float = PROBS["mutation"]
    # Parent has 2 bad genes. It'll try to pass on a bad gene unless mutation
    #   happens
    if parent_num_genes == 2:
        return 1-mutation_prob if bad_gene else mutation_prob
    # Parent has 1 bad gene. Whatever the case, the probability is 50%.
    #   This is because the mutation probability lost from one gene is
    #   complemented by the other gene with the opposite condition
    elif parent_num_genes == 1:
        return 0.5
    # Parent has 0 bad genes. It'll try to pass on a good gene unless mutation
    #   happens
    else:
        return mutation_prob if bad_gene else 1-mutation_prob


def update(probabilities: Dict[str, Dict[str, Dict]],
           one_gene: Set[str], two_genes: Set[str],
           have_trait: Set[str],
           p: float) -> None:
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        num_genes = get_num_genes(person, one_gene, two_genes)
        has_trait = person in have_trait

        # Update `gene`
        probabilities[person]["gene"][num_genes] += p

        # Update `trait`
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities: Dict[str, Dict[str, Dict]]) -> None:
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # Normalize `gene`
        gene_dict = probabilities[person]["gene"]
        gene_total = sum(gene_dict.values())
        probabilities[person]["gene"] = {n: gene_dict[n]/gene_total
                                         for n in gene_dict}

        # Normalize `trait`
        trait_dict = probabilities[person]["trait"]
        trait_total = sum(trait_dict.values())
        probabilities[person]["trait"] = {n: trait_dict[n]/trait_total
                                          for n in trait_dict}


if __name__ == "__main__":
    main()
