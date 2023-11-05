# Endgame Bot
A codebase for Stanisław Frejlak's master's thesis "Deep learning and combinatorial game theory; evaluating temperatures in the game of go".

## Installation

Clone this repository to your local machine. The repository uses one submodule: [my fork](https://github.com/siasio/a0-jax) of the `a0-jax` [repository](https://github.com/NTT123/a0-jax). If after cloning the current repository, the `a0-jax` directory remains empty, navigate to the current repository root and run: `git clone https://github.com/siasio/a0-jax`.

The project uses relative imports from the `a0-jax` submodule, so you need to make sure that they are recognized by Python. In PyCharm, you can achieve it by right-clicking on the directory and choosing `Mark Directory as: Sources Root`. In terminal, export the `a0-jax` directory path to the `PYTHONPATH`.

Find a fine-tuned Endgame Bot in the release section of this repository. Download it to the `a0-jax` directory.

To install libraries required for using the bot run:

`pip install -r vis_requirements.txt`

If your computer has a dedicated GPU, you could benefit from installing `jax` with GPU support. For Linux, follow instructions from the jax documentation [here](https://jax.readthedocs.io/en/latest/installation.html). Windows version can be installed from unofficial community builds [here](https://github.com/cloudhan/jax-windows-builder).

Note that the network is quite light so `jax` CPU version is enough for using it.

If you want to run the training loop by yourself, more libraries are required. Install them with:

`pip install -r train_requirements.txt`

For training, it's recommended to use a machine with a dedicated GPU, and to install `jax` with GPU support.

### Known issues

Issues might arise if you use versions of `optax` or `chex` other than specified in the requirements file. Make sure that you have the correct versions installed.

Windows installation of GPU supported `jax` version from the community builds might prove impossible on specific computers. In such case, use the CPU version.

In `train_requirements.txt`, `opencv-python-headless` is listed. If you have `opencv-python` instead, you might run into issues with the visualization tool because of a conflict between `PyQt` and `OpenCV` libraries.

## Usage

To use a GUI, run `visualize.py`. Press "Set up position" and put Black and White stones on the board. When "Black" is selected, you can add Black stones with mouse left-click, and White stones with right-click. When you set the position, press "Mask" and select intersections which count into the local position. After you press "Finish and analyze" a game tree for the local position will be created, and you can view it on the right-side panel. To the right of the panel some statistics will be displayed: 
 - a probability that White or Black gets the next move in the local position - ideally, we would like it to be 50-50 for positions with gote moves, 100-0 for positions with a sente move of one of the players, or if the previous move was sente, and 0-0 for finished positions
 - a local result calculated with Chinese rules, i.e. an expected difference between the number of Black's and White's intersections within the masked region
 - a value of move calculated with Japanese rules (which is twice the local temperature); something is incorrect when calculating values of reverse sente moves - should be revised

There is a basic code for managing kos (including multi-stage) which stops the algorithm from infinite tree expansion, and calculates a move value based on the initial and final position of a ko.
![Screenshot from 2023-11-01 20-03-23](https://github.com/siasio/EndgameBot/assets/39811817/8e7b4da2-6b2a-4ed6-9974-97939db5a516)

Currently the tree is expanded using very simple heuristics. In each position, only one move for black and one move for white are taken into account (the top choices of the neural network). To assess whether the network believes that the previous move is sente, I use a 10% threshold: if the sum of softmaxed moves of one color is below 10%, then I don't take that color into account when expanding the current node. Of course, this algorithm is imperfect, and it leads to wrong assessments from time to time. An example below shows a position in which a tree for a simple position is expanded way too much because of not detecting some sente moves. The currently displayed position after two moves should obviously be judged as Black's sente. However, Black's move probability being at 12%, the tree is expanded with both White's and Black's move:
![image](https://github.com/siasio/EndgameBot/assets/39811817/c8c67816-0bff-4fba-875c-74127f8f6dfa)

## Methods

The current version of the neural network was trained in the "knowledge distillation" setup, using KataGo ownership maps as the ground truth. To not train a network from scratch, I used a pre-trained AlphaZero network from the a0-jax repository. Since the AlphaZero network had already acquired a profound understanding of the game of go, fine-tuning it to make predictions about local positions was possible in 10 hours on a commodity hardware. As training data, I used hundreds of thousands of positions taken from KataGo self-play games.

I fine-tuned the network by discarding its original heads (which predict the winning chance, and a probability distribution over the next possible moves), and attaching new heads to it. Moreover, I concatenated the output of the last AlphaZero's hidden layer with a local position mask.

Below, I describe each of these steps in detail.

### Data collection

- I download KataGo self-play game recordings with `download_sgfs.py` script.
- Find the move number `n` of the first pass in the game (an indicator that the position is finished) and take a new move number `n' = 85% * n`. Taking a move with number `n'`, I get a position from an edgame stage of the game. The coefficient of 85% was found experimentally.
- On each of such positions, I run the KataGo network and store the ownership map predicted by it. Implementation of this and the previous step can be found in `sgf_to_position.py`. I utilize code from [KaTrain](https://github.com/sanderland/katrain/) to parse SGF files with game recordings, and run KataGo on the positions. I store the ownership maps in log files alongside with game recordings.
- Based on the ownership map, I break the game into areas which will almost certainly end as Black's territory, as White's territory, and areas for which the final owner is undecided. After setting a threshold of 95% for calling an intersection a part of a player's territory., I calculate the areas using `connectedComponents` function from OpenCV. The code can be found in `local_pos_masks.py`. The relevant methods are `board_segmentation` and `update_ownership`.
- Having decomposed the board into a sum of local undecided positions (and the sure territories), for each such local position I find the first move which was played within it. For each board position, I also pick one blob within a 5x5 square inside the sure territories area. I pickle the collected data: a 0-1 mask of a local position, an information about the next local, and the player at move at the current moment of the game. This step is implemented in `collect_ownership_data` inside `a0-jax/train_agent.py` script.

### Local position masks

To train a network which could make judgements about local positions, we obviously need to pass to it an information about the local position which it should look at. The AlphaZero network has a fixed input size, and adding a new channel to the input tensor is not possible. If one trains a network from scratch, then one could define the input tensor by themself, but in this case, it would be not possible to make use of the game understanding acquired throughout AlphaZero's training.

I take the third possible approach. I keep the backbone of AlphaZero network intact but I add an information about the local position to its final output. On top of it, I train a light network (i.e. the new heads). This approach assumes that a knowledge about local positions is present in the final hidden states of AlphaZero, and it could be easily revelead by fine-tuning. Noteworthy, in the fine-tuning process I keep the backbone weights frozen. (In fact, I unfreeze them at one point to make the network adjust even better to the new task, but I do it only at a point when the network's judgements are already quite accurate.)

To add the information about local position, I simply concatenate a local position mask with the output of AlphaZero's last hidden layer.

### Ownership head

The task of this head is to predict KataGo's ownership prediction of each intersection in the local position. The loss function takes only the masked intersections into account. However, after training it can be noticed that this head produces meaningful predictions also from other parts of the board, although not as accurate as for the masked region. I experimented also with using the final board position instead of KataGo's predictions as the ground truth, but I decided that it would only introduce unnecessary variance, without any clear benefits. As Katago predictions could be called more biased, the final position is also not what we aim at predicting, and it is introducing an unnatural two-class distinction contrary to KataGo's smooth predictions.

### Policy head

This head aims at predicting the next move: the intersection at which it was played as well as the color of the next move. There is one more possibility: that in a given local position no one played any move until the end of the game. I represent the target of this head as a one-dimensional tensor of legth 723 = 2 * 19 * 19 + 1. I experimented also with a spatially encoded representations (a three-dimensional tensor) but it led to a slower training.

## Further work

To be updated. The further work is to collect the actual Ground Truth data using the network which we have by now, and then retraining the network.
