# Endgame Bot
A codebase for Stanisław Frejlak's master's thesis "Deep learning and combinatorial game theory; evaluating temperatures in the game of go".

## Installation

Detailed instructions will be added soon. Requirements are listed in the `requirements.txt` file. Setting up the environment might be difficult on Windows because of `jax` library. To build `jax` under Windows, use this [repo](https://github.com/cloudhan/jax-windows-builder). Install `jaxlib` with CUDA to utilize your GPU. Make sure that you have required version of `opax`, `optax`, `chex`, `jax` and `jaxlib` libraries.

Make sure that you have `opencv-python-headless` and not `opencv-python` installed. Otherwise, a GUI might not run.

## Usage

Find a fine-tuned Endgame Bot in the realese section of this repository. Download it to the `a0-jax` directory.

To use a GUI, run `goban.py`. Press "Set up position" and put Black and White stones on the board. When "Black" is selected, you can add Black stones with mouse left-click, and White stones with right-click. When you set the position, press "Mask" and select intersections which count into the local position. After you press "Finish and analyze" a game tree for the local position will be created, and you can view it on the right-side panel. To the right of the panel some statistics will be displayed: 
 - a probability that White or Black gets the next move in the local position - ideally, we would like it to be 50-50 for positions with gote moves, 100-0 for positions with a sente move of one of the players, or if the previous move was sente, and 0-0 for finished positions
 - a local result calculated with Chinese rules, i.e. an expected difference between the number of Black's and White's intersections within the masked region
 - a value of move calculated with Japanese rules (which is twice the local temperature); something is incorrect when calculating values of reverse sente moves - should be revised

![Screenshot from 2023-11-01 20-03-23](https://github.com/siasio/EndgameBot/assets/39811817/8e7b4da2-6b2a-4ed6-9974-97939db5a516)
![Screenshot from 2023-11-01 20-03-51](https://github.com/siasio/EndgameBot/assets/39811817/fcb77236-16cf-4957-9305-700abe04bba1)
![Screenshot from 2023-11-01 20-04-04](https://github.com/siasio/EndgameBot/assets/39811817/155b21c6-009a-4f2d-840f-ea723a6e2927)
