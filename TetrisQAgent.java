package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        final int hiddenDim = 4 * numPixelsInImage; // can change this dim 
        final int outDim = 1;

        Sequential qFunction = new Sequential();

        // add input layer
        qFunction.add(new Dense(numPixelsInImage, hiddenDim));
        qFunction.add(new ReLU()); //changed to ReLU activation fxn -> can try ReLU, Tanh, Sigmoid
        // add output layer
        qFunction.add(new Dense(hiddenDim, outDim));

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        // get some game state info
        Board board = game.getBoard();
        Mino currentMino = game.getCurrentMino();
        int totalScore = game.getTotalScore();
        int scoreThisTurn = game.getScoreThisTurn();
        List<Mino.MinoType> nextThreeMinoTypes = game.viewNextThreeMinoTypes();
        boolean isTSpin = game.wasTSpin(currentMino);
        boolean isDoubleTSpin = game.wasDoubleTSpin(currentMino);
        boolean isGameOver = game.isOver();
        Map<Mino, Game.BFSNode> finalMinoPositions = game.getFinalMinoPositions();

        // make input vector
        int inputSize = calculateInputSize(board);
        Matrix inputVector = new Matrix(1, inputSize);
        int idx = 0;

        // encode board state into input vector
        encodeBoardState(inputVector, board);

        // add other game state features to the input vector
        inputVector.set(0, idx++, totalScore);
        inputVector.set(0, idx++, scoreThisTurn);
        inputVector.set(0, idx++, nextThreeMinoTypes.size());
        inputVector.set(0, idx++, isTSpin ? 1 : 0);
        inputVector.set(0, idx++, isDoubleTSpin ? 1 : 0);
        inputVector.set(0, idx++, isGameOver ? 1 : 0);

        // normalize input features (I think this will help with converging faster)
        normalizeInput(inputVector);

        return inputVector;
    }

    // helper method
    private int calculateInputSize(Board board) {
        int numRows = board.getNumRows();
        int numCols = board.getNumCols();
        return numRows * numCols + 6; // include additional game state features
    }

    // helper method to encode board state into input vector
    private void encodeBoardState(Matrix inputVector, Board board) {
        int numRows = board.getNumRows();
        int numCols = board.getNumCols();
        int[][] boardMatrix = board.toArray();
        int idx = 0;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                inputVector.set(0, idx++, boardMatrix[i][j]);
            }
        }
    }

    // helper method
    private void normalizeInput(Matrix inputVector) {
        double maxFeatureValue = inputVector.max();
        double minFeatureValue = inputVector.min();
        double range = maxFeatureValue - minFeatureValue;
        for (int i = 0; i < inputVector.getColumns(); i++) {
            double normalizedValue = (inputVector.get(0, i) - minFeatureValue) / range;
            inputVector.set(0, i, normalizedValue);
        }
    }

    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        return random.nextDouble() < EXPLORATION_PROB; // Explore with probability EXPLORATION_PROB
    }

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());
        return game.getFinalMinoPositions().get(randIdx);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game)
    {
        double reward = game.getScoreThisTurn(); // start with this turns score
        int maxHeight = game.getMaxHeight(); // getMaxHeight() is just a place holder rn, not defined i dont think
        reward -= maxHeight; // penalizes higher stack height
        reward += game.getLinesCleared(); // rewards for lines cleared
        return reward;
    }

}

