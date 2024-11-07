"""
store all the agents here
"""
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DQNModel(nn.Module):
    '''
    class to read the version_config file and initialize pytorch CNN based on it.
    Used in DeepQLearningAgent(Agent)'s _agent_model(self) method
    '''
    def __init__(self, config_path, board_size, n_frames, n_actions):
        super(DQNModel, self).__init__()
        self.n_actions = n_actions
        self.layers = nn.ModuleList()

        #read the configuration file
        with open(config_path, 'r') as f:
            config = json.load(f)

        #Create input layer shape based on board size and frames
        input_shape = (n_frames, board_size, board_size)

        #Define layers based on JSON configuration
        for layer_name, params in config["model"].items():
            if 'Conv2D' in layer_name:
                conv_layer = nn.Conv2d(
                    in_channels=input_shape[0],  #set in_channels as n_frames
                    out_channels=params["filters"],
                    kernel_size=tuple(params["kernel_size"]),
                    stride=params.get("strides", (1, 1)),
                    padding=params.get("padding", 0),
                )
                self.layers.append(conv_layer)
                #Activation func:
                self.layers.append(nn.ReLU())
                input_shape = (params["filters"],  #update in_channels for the next layer
                               input_shape[1] - params["kernel_size"][0] + 1,  #height
                               input_shape[2] - params["kernel_size"][1] + 1) #width

            elif 'Flatten' in layer_name:
                self.layers.append(nn.Flatten())
                input_shape = input_shape[0] * input_shape[1] * input_shape[2] *4  # Flattened size
                print(input_shape)

            elif 'Dense' in layer_name:
                dense_layer = nn.Linear(input_shape[0] * input_shape[1] * input_shape[2] if isinstance(input_shape, tuple) else input_shape, 
                                        params["units"])
                self.layers.append(dense_layer)
                self.layers.append(nn.ReLU())

        # Final output layer for action values
        self.output_layer = nn.Linear(params["units"], n_actions)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            print(x.shape)
        return self.output_layer(x)

class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    
    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col

class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values
    This class extends to the following classes
    PolicyGradientAgent
    AdvantageActorCriticAgent

    Attributes
    ----------
    _model : TensorFlow Graph
        Stores the graph of the DQN model
    _target_net : TensorFlow Graph
        Stores the target network graph of the DQN model
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.reset_models()

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model, self.optimizer = self._agent_model()
        self._model.to(device)
        self.loss = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        if(self._use_target_net):
            self._target_net,__ = self._agent_model()
            self._target_net.to(device)
            self.update_target_net()

   # def _prepare_input(self, board):
   #     """Reshape input and normalize
   #     
   #     Parameters
   #     ----------
   #     board : Numpy array
   #         The board state to process
#
   #     Returns
   #     -------
   #     board : Numpy array
   #         Processed and normalized board
   #     """
   #     if(board.ndim == 3):
   #         board = board.reshape((1,) + self._input_shape)
   #     board = self._normalize_board(board.clone())
   #     return board.clone()
    def _prepare_input(self, board):
        """Reshape input and normalize. Parameters
                ----------
        oard : Tensor
        The board state to process, in shape [batch_size, height, width, channels]

    Returns
    -------
    board : Tensor
        Processed and normalized board in shape [batch_size, channels, height, width]
    """
        if isinstance(board, np.ndarray):
            board = torch.tensor(board, dtype=torch.float32).to(device)  # Use the correct device
        
        if board.ndim == 4:
        # Change from [batch_size, height, width, channels] to [batch_size, channels, height, width]
            board = board.permute(0, 3, 1, 2)
    
            board = self._normalize_board(board)
        return board


    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : TensorFlow Graph, optional
            The graph to use for prediction, model or target network

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board, 
            of shape board.shape[0] * num actions
        """
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
            model_outputs = model(board)
        else:
            
            model.eval()  #set to evaluation mode
            with torch.no_grad():  #disable gradient calculations
                model_outputs = model(board)

        return model_outputs

    def _normalize_board(self, board):
        """Normalize the board before input to the network
        
        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        # return board.copy()
        # return((board/128.0 - 1).copy())
        return board.to(dtype=torch.float32) / 4.0

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        model_outputs_cpu = model_outputs.cpu().detach().numpy()  # Move to CPU and detach from computation graph

        
        
        return np.argmax(np.where(legal_moves == 1, model_outputs_cpu, -np.inf), axis=1)

    def _agent_model(self):
        # Load the model config and initialize DQNModel
        config_path = 'model_config/{:s}.json'.format(self._version)
        model = DQNModel(config_path, self._board_size, self._n_frames, self._n_actions)
    

        optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
    
        return model, optimizer
        


    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Save the current model weights
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
            
            
        #save the model weights (state_dict) instead of the entire model
        torch.save(self._model.state_dict(), "{}/model_{:04d}.pth".format(file_path, iteration))
    
        if self._use_target_net:
            torch.save(self._target_net.state_dict(), "{}/model_{:04d}_target.pth".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using pytorch's
        inbuilt load state dict function (model saved in .pth format)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        #load the model weights (state_dict) from file
        model_weights = torch.load("{}/model_{:04d}.pth".format(file_path, iteration))
        self._model.load_state_dict(model_weights)
    
        if self._use_target_net:
            target_net_weights = torch.load("{}/model_{:04d}_target.pth".format(file_path, iteration))
            self._target_net.load_state_dict(target_net_weights)

    def print_models(self):
        """Print the current models using summary method"""
        print('Training Model')
        print(summary(self._model))
        if(self._use_target_net):
            print('Target Network')
            print(summary(self._target_net))

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        We are predicting the expected future discounted reward for all
        actions with our model. The target for training the model is calculated
        in two parts:
        1) dicounted reward = current reward + 
                        (max possible reward in next state) * gamma
           the next reward component is calculated using the predictions
           of the target network (for stability)
        2) rewards for only the action take are compared, hence while
           calculating the target, set target value for all other actions
           the same as the model predictions
        
        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
            The current error (error metric is defined in reset_models)
        """
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if(reward_clip):
            r = np.sign(r)
        
        s = torch.tensor(s, dtype=torch.float32).to(device)
        a = torch.tensor(a, dtype=torch.float32).to(device)
        r = torch.tensor(r, dtype=torch.float32).to(device)
        next_s = torch.tensor(next_s, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)
        legal_moves = torch.tensor(legal_moves, dtype=torch.float32).to(device)
        print("r shape: ", r.shape)
        print("a shape: ", a.shape)
        print("s shape: ", s.shape)
        print("next_S shape: ", next_s.shape)
        print("done shape: ", done.shape)
        print("legal moves shape: ", legal_moves.shape)
        
        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        # Get outputs for next state
        next_model_outputs = self._get_model_outputs(next_s, current_model)
        # our estimate of expexted future discounted reward
        
        # Calculate discounted reward
        max_next_reward = torch.max(torch.where(legal_moves == 1, next_model_outputs, -torch.inf), dim=1)[0]
        max_next_reward = max_next_reward.unsqueeze(1)
        
        discounted_reward = r + (self._gamma * max_next_reward * (1 - done))
        
        # create the target variable, only the column with action has different value
        target = self._get_model_outputs(s, current_model)
        
        # we bother only with the difference in reward estimate at the selected action
        target = (1-a)*target + a*discounted_reward
        
        self._model.train()
        
        #Getting action values from the training net, not the target net
        model_output = self._get_model_outputs(s)
        # Make sure model_output requires gradients
        model_output.requires_grad_()
        
        # mean huber loss
        loss = self.loss(model_output, target)
        # Zero gradients, perform backward pass, and update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if self._use_target_net:
        #copy the weights from the main model to the target network
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        """Utility function to check if the model
        and target network have the same weights or not"""
        
        if self._use_target_net:
            for i, (model_param, target_param) in enumerate(zip(self._model.parameters(), self._target_net.parameters())):
            # Check if the weights are equal
                weights_match = torch.equal(model_param, target_param)
                print(f'Parameter {i} Match: {weights_match}')
                
                

    def copy_weights_from_agent(self, agent_for_copy):
        """Update weights between competing agents which can be usedin parallel training"""
        assert isinstance(agent_for_copy, type(self)), "Agent type is required for copy"

        # Copy weights from agent_for_copy's model and target network to this agent's model and target network
        self._model.load_state_dict(agent_for_copy._model.state_dict())
        if self._use_target_net:
            self._target_net.load_state_dict(agent_for_copy._target_net.state_dict())






