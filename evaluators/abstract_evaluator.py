import copy
import json
from typing import List, Union

import numpy as np

from sgf_utils.game_node import GameNode


class AbstractEvaluator:
    def __init__(self):
        self.eval_queue = []
        self.inference_count = 0

    def get_key(self, position: GameNode, mask: np.ndarray) -> str:
        # Consider implementing it here
        # It will probably rarely differ in different evaluators
        # TODO: Think whether KataGo evaluator should ignore mask as a part of the key
        # Probably it cannot and needs an internal registry to avoid running inference for the same position many times
        raise NotImplementedError

    def set_black_moves(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        return self.set_moves_for_color(evaluation, 'B')

    def set_white_moves(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        return self.set_moves_for_color(evaluation, 'W')

    def set_moves_for_color(self, evaluation: Union[List['Evaluation'], 'Evaluation'], color: str):
        raise NotImplementedError

    def set_black_next_prob(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        raise NotImplementedError

    def set_no_move_prob(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        raise NotImplementedError

    def set_ownership(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        raise NotImplementedError

    def set_score(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        raise NotImplementedError

    def set_predicted_mask(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        raise NotImplementedError

    def add_to_queue(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        return
        if isinstance(evaluation, Evaluation):
            if evaluation not in self.eval_queue:
                self.eval_queue.append(evaluation)
        elif isinstance(evaluation, (list, tuple)):
            self.eval_queue.extend([ev for ev in evaluation if ev not in self.eval_queue])
        else:
            raise ValueError("Expected Evaluation or a list of Evaluations")

    def shutdown(self):
        del self

    def reset(self):
        self.eval_queue = []
        self.inference_count = 0


class Evaluation:
    def __init__(self, evaluator: AbstractEvaluator, position: GameNode, mask: np.ndarray):
        """
        Evaluation stores evaluator's output for a position:
        predictions for the next local moves for both colors
        information which color is more likely to get the next move (expressed as probability)
        information whether this is a terminal position (expressed as probability)
        expected ownership of each intersection
        expected local score
        predicted mask of a local position based on the input mask

        Different evaluators may be able to evaluate some, or all, of these pieces of information.

        The pieces of information are stored in private variables (starting with underscore)
        and they are supposed to be accessed via the non-private properties (version with no underscore)
        """
        self.evaluator = evaluator
        self.position = position
        self.mask = mask

        self._black_moves: np.ndarray = None
        self._white_moves: np.ndarray = None
        self._black_prob: float = None
        self._no_move_prob: float = None
        self._ownership: np.ndarray = None
        self._score: float = None  # from Black's (L's) perspective
        self._predicted_mask: np.ndarray = None
        self.quantize_ownership = True  # It is a bit not elegant to specify it here

    def __deepcopy__(self, memodict={}):
        new_obj = type(self)(self.evaluator, self.position, self.mask)
        return new_obj

    def to_json(self):
        return json.dumps({
            'black_moves': self._black_moves.tolist(),
            'white_moves': self._white_moves.tolist(),
            'black_prob': self._black_prob,
            'no_move_prob': self._no_move_prob,
            'ownership': self._ownership.tolist(),
            'score': self._score,
            'predicted_mask': self._predicted_mask.tolist(),
        })

    @property
    def moves(self):
        return [self.black_moves, self.white_moves]

    @property
    def black_moves(self):
        if self._black_moves is None:
            self.evaluator.set_black_moves(self)
        return self._black_moves

    @black_moves.setter
    def black_moves(self, value: np.ndarray):
        # Consider warning if it is already set to a different value
        self._black_moves = value

    @property
    def white_moves(self):
        if self._white_moves is None:
            self.evaluator.set_white_moves(self)
        return self._white_moves

    @white_moves.setter
    def white_moves(self, value: np.ndarray):
        self._white_moves = value

    @property
    def black_prob(self):
        if self._black_prob is None:
            self.evaluator.set_black_next_prob(self)
        return self._black_prob

    @black_prob.setter
    def black_prob(self, value: float):
        self._black_prob = value

    @property
    def white_prob(self):
        return 1 - self.black_prob

    @white_prob.setter
    def white_prob(self, value: float):
        self._black_prob = 1 - value

    @property
    def no_move_prob(self):
        if self._no_move_prob is None:
            self.evaluator.set_no_move_prob(self)
        return self._no_move_prob

    @no_move_prob.setter
    def no_move_prob(self, value: float):
        self._no_move_prob = value

    @property
    def ownership(self):
        if self._ownership is None:
            self.evaluator.set_ownership(self)
        return self._ownership

    @ownership.setter
    def ownership(self, value):
        self._ownership = value

    @property
    def score(self):
        if self._score is None:
            self.evaluator.set_score(self)
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def predicted_mask(self):
        if self._predicted_mask is None:
            self.evaluator.set_predicted_mask(self)
        return self._predicted_mask

    @predicted_mask.setter
    def predicted_mask(self, value):
        self._predicted_mask = value


def unpack_list_decorator(method):
    def wrapper(self, obj):
        if isinstance(obj, list):
            results = []
            for item in obj:
                results.append(method(self, item))
            return results
        else:
            return method(self, obj)
    return wrapper


def pack_single_decorator(method):
    def wrapper(self, obj):
        if not isinstance(obj, list):
            obj = [obj]
        results = method(self, obj)
        if isinstance(results, list) and len(results) == 1:
            return results[0]
        return results
    return wrapper
