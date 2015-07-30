"""
Module for storing concept graphs

@author Siddharth Reddy <sgr45@cornell.edu>
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

        :param dict[str,set[str]] concepts_of_module:
            A dictionary mapping module id to a set of concept ids

        :param dict[str,set[str]] prereqs_of_concept:
            A dictionary mapping concept id to a set of prereq concept ids
        """

        self.concepts_of_module = concepts_of_module
        self.prereqs_of_concept = prereqs_of_concept

        concept_ids = set(self.prereqs_of_concept.keys())
        concept_ids |= {c for v in self.prereqs_of_concept.values() for c in v}
        concept_ids |= {c for v in self.concepts_of_module.values() for c in v}

        self.idx_of_concept_id = {k: i for i, k in enumerate(concept_ids)}
        self.num_concepts = len(self.idx_of_concept_id)

    def concept_module_edges(self, iter_modules, idx_of_module_id):
        """
        Get a list of concept-module edges in the graph

        :param function iter_modules: :py:func:`datatools.InteractionHistory.iter_assessments`
            or :py:func:`InteractionHistory.iter_lessons`
        
        :param function idx_of_module_id: 
            :py:func:`datatools.InteractionHistory.idx_of_assessment_id` or 
            :py:func:`datatools.InteractionHistory.idx_of_lesson_id`

        :rtype: (np.array,np.array,int,int,np.array)
        :return: A tuple of (module indexes, concept indexes,
            number of unique modules, number of unique concepts,
            number of concepts for each module in the first array of this tuple)
        """

        module_ids, concept_ids = zip(*[(module_id, concept_id)
            for module_id in iter_modules()
            for concept_id in self.concepts_of_module.get(module_id, [])])

        num_concepts_of_modules = np.array(
            [len(self.concepts_of_module[module_id]) for module_id in module_ids])

        num_modules = sum(1 for _ in iter_modules())

        module_idxes = np.array([idx_of_module_id(module_id) for module_id in module_ids])
        concept_idxes = np.array(
                [self.idx_of_concept_id[concept_id] for concept_id in concept_ids])

        return (
            module_idxes, concept_idxes,
            num_modules, self.num_concepts,
            num_concepts_of_modules)

    def concept_prereq_edges(self):
        """
        Get a pair of lists, the first of which represents the heads of all prereq edges (i.e., the
        prerequisite concepts) and the second of which represents the tails of all prereq edges
        (i.e., the postrequisite concepts). That is, the pair (first_list[i], second_list[i]) is
        an edge in the graph with the first element being the prerequisite and the second element
        being the postrequisite.

        Also returns the number of concepts in the graph for bookkeeping purposes.

        :rtype: (np.array,np.array,int)
        :return: A tuple of (prereq_concept_idxes, postreq_concept_idxes, num_concepts)
        """

        prereq_ids = []
        postreq_ids = []
        for postreq_id, my_prereq_ids in self.prereqs_of_concept.iteritems():
            postreq_ids.extend([postreq_id] * len(my_prereq_ids))
            prereq_ids.extend(my_prereq_ids)

        prereq_idxes = np.array([self.idx_of_concept_id[concept_id] for concept_id in prereq_ids])
        postreq_idxes = np.array(
                [self.idx_of_concept_id[concept_id] for concept_id in postreq_ids])

        return (prereq_idxes, postreq_idxes, self.num_concepts)

