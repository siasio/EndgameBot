import json
import os
import time

from tqdm import tqdm
from typing import Dict
# import yaml
from goban import GoBoard, QApplication, MainWindow

RESPONSE_STR = 'Response: '
REQUEST_STR = 'Request: '

OPPOSITE_COLOR = {'W': 'B', 'B': 'W'}


def get_base_id(pos_id):
    # The IDs have form G{number_of_game}M{number_of_move}{player} where player is 'B' or 'W'
    return pos_id[:-1]


def get_short_id(pos_id):
    # The IDs have form G{number_of_game}M{number_of_move}{player} where player is 'B' or 'W'
    return pos_id.split('M')[0]


class RequestWithResponse:
    def __init__(self):
        self.request_dict = None
        self.w_response_dict = None
        self.b_response_dict = None
        self.registered_w_request, self.registered_b_request = False, False
        self.request_color_mapping = {'B': self.registered_b_request, 'W': self.registered_w_request}
        # Note that the request_color_mapping dictionary points to boolean attributes
        # The response color mapping points to dictionary containing the responses
        self.response_color_mapping = {'B': self.b_response_dict, 'W': self.w_response_dict}

    def register_line(self, line_dict):
        if 'ownership' in line_dict:
            self.register_response(line_dict)
        else:
            self.register_request(line_dict)

    def register_request(self, request_dict):
        color = request_dict['initialPlayer']
        assert self.request_color_mapping[color] is False, f'{self.pos_id}: repeated request for color {color}'
        assert request_dict['id'][-1] == color, f'{self.pos_id}: In request for color {color}, the id ({request_dict["id"]}) doesn\'t follow the convention'
        if self.request_color_mapping[OPPOSITE_COLOR[color]] is False:
            self.request_dict = request_dict
        else:
            old_black_stones = set(self.stones['B'])
            old_white_stones = set(self.stones['W'])
            self.request_dict = request_dict
            black_stones = set(self.stones['B'])
            white_stones = set(self.stones['W'])
            assert black_stones == old_black_stones and white_stones == old_white_stones, f'Positions don\'t match for black and white requests for {self.pos_id}'
        if color == 'B':
            self.registered_b_request = True
        else:
            self.registered_w_request = True

    def register_response(self, response_dict):
        color = response_dict['id'][-1]
        assert color in 'BW', f'{self.pos_id}: In a response, the id ({response_dict["id"]}) doesn\'t follow the convention'
        assert self.response_color_mapping[color] is None, f'{self.pos_id}: repeated response for color {color}'
        if color == 'B':
            self.b_response_dict = response_dict
        else:
            self.w_response_dict = response_dict

    # @property
    # def player(self):
    #     return None if self.request_dict is None else self.request_dict['initialPlayer']

    @property
    def is_final(self):
        return 'last' in self.pos_id

    @property
    def completed(self):
        if not self.is_final:
            return self.registered_w_request and self.registered_b_request and self.registered_b_response and self.registered_w_response

        return (self.registered_w_request and self.registered_w_response) or (self.registered_b_request and self.registered_b_response)

    @property
    def registered_w_response(self):
        return self.w_response_dict is not None

    @property
    def registered_b_response(self):
        return self.b_response_dict is not None

    @property
    def stones(self):
        if self.request_dict is None:
            return None

        stones = self.request_dict['initialStones']
        black_stones = []
        white_stones = []
        for stone in stones:
            if stone[0] == 'B':
                black_stones.append(stone[1])
            elif stone[0] == 'W':
                white_stones.append(stone[1])
            else:
                raise KeyError(f'Stone {stone} in {self.pos_id} has no color specified!')
        return {'B': black_stones, 'W': white_stones}
        # return None if self.request_dict is None else self.request_dict['initialStones']

    @property
    def moves(self):
        return None if self.request_dict is None else self.request_dict['moves']

    @property
    def board_size(self):
        return None if self.request_dict is None else (self.request_dict['boardXSize'], self.request_dict['boardYSize'])

    @property
    def w_ownership(self):
        return None if self.w_response_dict is None else self.w_response_dict['ownership']

    @property
    def b_ownership(self):
        return None if self.b_response_dict is None else self.b_response_dict['ownership']

    @property
    def w_score(self):
        return None if self.w_response_dict is None else self.w_response_dict['rootInfo']['scoreLead']

    @property
    def b_score(self):
        return None if self.b_response_dict is None else self.b_response_dict['rootInfo']['scoreLead']

    @property
    def pos_id(self):
        if self.request_dict is not None:
            return get_base_id(self.request_dict['id'])
        elif self.w_response_dict is not None:
            return get_base_id(self.w_response_dict['id'])
        elif self.b_response_dict is not None:
            return get_base_id(self.b_response_dict['id'])
        return None

    def merged_dict(self):
        # not listing moves as in the current use case we have no moves
        b_scaled = self.scale_ownerships(self.b_ownership)
        w_scaled = self.scale_ownerships(self.w_ownership)
        merged = {'id': self.pos_id, 'size': self.board_size, 'stones': self.stones}
        if b_scaled:
            merged['b_own'] = b_scaled
        if w_scaled:
            merged['w_own'] = w_scaled
        if self.b_score:
            merged['b_score'] = f'{self.b_score:.2f}'
        if self.w_score:
            merged['w_score'] = f'{self.w_score:.2f}'
        return merged

    @staticmethod
    def scale_ownerships(ownership_list):
        """

        :param ownership_list: list of floats between -1 and 1
        :return: natural numbers between 0 and 100
        """
        if ownership_list is None:
            return None
        return [round(50 * (ownership + 1)) for ownership in ownership_list]

    def __repr__(self):
        return json.dumps(self.merged_dict())


class PerGameRequestWithResponse:
    def __init__(self, short_id=''):
        self.requests_dict: Dict[str, RequestWithResponse] = {}
        self.short_id = short_id
        self.sgf = ''

    def add_request_response(self, request_response: RequestWithResponse):
        short_id = get_short_id(request_response.pos_id)
        assert short_id == self.short_id, f'IDs don\'t match. Trying to register {short_id} instead of {self.short_id}'
        self.requests_dict[request_response.pos_id] = request_response

    def add_sgf(self, sgf):
        assert len(self.sgf) == 0, f'Sgf being registered more than once for {self.short_id}'
        self.sgf = sgf

    def __repr__(self):
        jsons_dict = {key: self.requests_dict[key].merged_dict() for key in self.requests_dict}
        jsons_dict['id'] = self.short_id
        jsons_dict['sgf'] = self.sgf
        return json.dumps(jsons_dict)


# RequestsResponsesDict = TypedDict('RequestsResponsesDict', {'id': RequestWithResponse})


def simplify_log(filepath, new_filepath, sgfs_path):
    used_ids = set()
    requests_with_responses_dict: Dict[str, RequestWithResponse] = {}
    per_game_requests_responses: Dict[str, PerGameRequestWithResponse] = {}
    if not os.path.exists(filepath):
        time.sleep(5)
    if not os.path.exists(filepath):
        print(f'{filepath} doesn\'t exist! Aborting simplifying log')
        return
    with open(filepath, 'r') as log_file:
        with open(sgfs_path, 'r') as sgfs:
            sgfs_dict = {f'G{i}': sgf for i, sgf in enumerate(sgfs.read().splitlines(), 1)}
            for l in tqdm(log_file):
                l = l.strip()
                if REQUEST_STR in l or RESPONSE_STR in l:
                    assert not (REQUEST_STR in l and RESPONSE_STR in l), f'Strange line {l}'
                    line_split = l.split(REQUEST_STR)[-1].split(RESPONSE_STR)[-1]
                    # assert len(line_split) == 2, f'{filepath}: Found {len(line_split) - 1} substrings "{REQUEST_STR}" in {l}'
                    cur_dict = json.loads(line_split)
                    pos_id = cur_dict['id']
                    base_id = get_base_id(pos_id)
                    short_id = get_short_id(pos_id)
                    try:
                        cur_sgf = sgfs_dict.pop(short_id)
                        try:
                            cur_per_game_request_response = per_game_requests_responses[short_id]
                        except KeyError:
                            cur_per_game_request_response = PerGameRequestWithResponse(short_id)
                            per_game_requests_responses[short_id] = cur_per_game_request_response
                        finally:
                            cur_per_game_request_response.add_sgf(cur_sgf)
                    except KeyError:
                        # the sgf string was already popped and registered
                        pass
                    line_type = 'rq' if REQUEST_STR in l else 'rsp'
                    used_id = f'{line_type}:{pos_id}'
                    if used_id in used_ids:
                        print(f'{filepath}: Found two rows {used_id}')
                        continue
                        # input()
                    # assert used_id not in used_ids, f'Found two rows {used_id}'
                    used_ids.add(used_id)
                    try:
                        cur_request_response = requests_with_responses_dict[base_id]

                    except KeyError:
                        cur_request_response = RequestWithResponse()
                        requests_with_responses_dict[base_id] = cur_request_response

                    finally:
                        cur_request_response.register_line(cur_dict)
                        if cur_request_response.completed:
                            cur_request_response = requests_with_responses_dict.pop(base_id)
                            try:
                                cur_per_game_request_response = per_game_requests_responses[short_id]
                            except KeyError:
                                cur_per_game_request_response = PerGameRequestWithResponse(short_id)
                                per_game_requests_responses[short_id] = cur_per_game_request_response
                            finally:
                                cur_per_game_request_response.add_request_response(cur_request_response)
                            # new_file.write(f'{cur_request_response}\n')
                            # del cur_request_response

        with open(new_filepath, 'w') as new_file:
            for per_game in per_game_requests_responses.values():
                new_file.write(f'{per_game}\n')
                del per_game

        # print('Requests with responses dict:', requests_with_responses_dict)
        print(f'Registered {len(used_ids)} lines with responses/requests from {len(per_game_requests_responses)} games')

# simplify_log(r'analysis_logs/b15c192-s74759936-d68801237/B8F1EB31610F7EE2.log', r'refined_logs/b15c192-s74759936-d68801237/B8F1EB31610F7EE2.log')
log_dir = 'analysis_logs'
refined_log_dir = 'refined_logs'
# for log_filename in os.path.listdir(log_dir):
# # log_filename = '20230523-182327-41FDB589.log'
#     log_filepath = os.path.join(log_dir, log_filename)
#     refined_log_filepath = os.path.join(refined_log_dir, log_filename)
#     if not os.path.exists(refined_log_filepath):
#         read_log(log_filepath, refined_log_filepath)
# with open(refined_log_filepath, 'r') as f:
    # positions = f.read()
# app = QApplication([])
# window = MainWindow()
# window.show()
# app.exec_()
# counter = 0
# for pos in f:
#     pos = pos.strip()
#     pos = json.loads(pos)
#     window.go_board.stones_from_gtp(pos['stones'])
#     if counter == 0:
#         break



