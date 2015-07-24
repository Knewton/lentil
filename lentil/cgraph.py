"""
Module for storing concept graphs
@author Siddharth Reddy <sgr45@cornell.edu>
07/21/15
"""

import numpy as np


class ConceptGraph(object):
    """
    Class for storing a content-to-concept graph with an
    accompanying dependency graph for concepts
    """

    def __init__(self, concepts_of_module, prereqs_of_concept):
        """
        Initialize concept graph object

        :param dict[str, set[str]] concepts_of_module:
            A dictionary mapping module id to set(concept ids)

        :param dict[str, set[str]] prereqs_of_concept:
            A dictionary mapping concept id to set(prereq concept ids)
        """

        self.concepts_of_module = concepts_of_module
        self.prereqs_of_concept = prereqs_of_concept

        concept_ids = set(
            self.prereqs_of_concept.keys()) | {c for v in self.prereqs_of_concept.values(
            ) for c in v} | {c for v in self.concepts_of_module.values() for c in v}

        self.idx_of_concept_id = {k: i for i, k in enumerate(concept_ids)}
        self.num_concepts = len(self.idx_of_concept_id)

    def concept_module_edges(self, iter_modules, idx_of_module_id):
        """
        Get a list of concept-module edges in the graph

        :param function iter_modules: InteractionHistory.iter_assessments
            or InteractionHistory.iter_lessons
        :param function idx_of_module_id: InteractionHistory.idx_of_assessment_id
            or InteractionHistory.idx_of_lesson_id
        :rtype: (np.array, np.array, int, int, np.array)
        :return: A tuple of (module indexes, concept indexes,
            number of unique modules, number of unique concepts,
            number of concepts for each module in the first array of this tuple)
        """

        module_ids, concept_ids = zip(*[(
            module_id, concept_id) for module_id in iter_modules(
            ) for concept_id in self.concepts_of_module.get(module_id, [])])

        num_concepts_of_modules = np.array([len(
            self.concepts_of_module[module_id]) for module_id in module_ids])
        num_modules = sum(1 for _ in iter_modules())

        module_idxes = np.array([idx_of_module_id(
            module_id) for module_id in module_ids])
        concept_idxes = np.array(
            [self.idx_of_concept_id[concept_id] for concept_id in concept_ids])

        return (
            module_idxes, concept_idxes,
            num_modules, self.num_concepts,
            num_concepts_of_modules)

    def concept_prereq_edges(self):
        """
        Get a list of concept-concept prereq edges in the graph

        :rtype: (np.array, np.array, int)
        :return: A tuple of (prereq_concept_idxes, postreq_concept_idxes, num_concepts)
        """

        prereq_ids = []
        postreq_ids = []
        for postreq_id, my_prereq_ids in self.prereqs_of_concept.iteritems():
            postreq_ids.extend([postreq_id] * len(my_prereq_ids))
            prereq_ids += my_prereq_ids

        prereq_idxes = np.array(
            [self.idx_of_concept_id[concept_id] for concept_id in prereq_ids])
        postreq_idxes = np.array(
            [self.idx_of_concept_id[concept_id] for concept_id in postreq_ids])

        return (prereq_idxes, postreq_idxes, self.num_concepts)
