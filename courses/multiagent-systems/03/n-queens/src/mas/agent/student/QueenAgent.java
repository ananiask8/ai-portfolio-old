package mas.agent.student;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import cz.agents.alite.communication.content.Content;
import mas.agent.MASQueenAgent;
import cz.agents.alite.communication.Message;

public class QueenAgent extends MASQueenAgent {
	class Position {
		private int row;
		private int col;

		public Position(int row, int col) {
			this.row = row;
			this.col = col;
		}

		public int getRow() { return this.row; }
		public int getCol() { return this.col; }
		public void setPosition(int row, int col) {
			this.row = row;
			this.col = col;
		}
		public boolean matchingRows(Position xj) { return row == xj.getRow(); }
		public boolean matchingColumns(Position xj) { return col == xj.getCol(); }
		public boolean matchingDiagonals(Position xj) { return Math.abs(row - xj.getRow()) == Math.abs(col - xj.getCol()); }
		@Override
		public String toString() { return "(" + row + ", " + col + ")"; }
		public boolean equals(Position xj) {
			return row == xj.getRow() && col == xj.getCol();
		}
	}
	class NoGood {
	    private int senderId;
	    private HashMap<Integer, Position> constraints;
	    public NoGood(int senderId) { this.senderId = senderId; constraints = new HashMap<>(); }
	    public NoGood(int senderId, HashMap<Integer, Position> constraints) {
	        this.senderId = senderId;
	        this.constraints = constraints;
        }
        public NoGood(int senderId, Position senderValue, Position constraint) {
            this.senderId = senderId;
            this.constraints = new HashMap<>();
            this.constraints.put(constraint.getRow(), constraint);
        }
        public void addConstraint(Position constraint) { this.constraints.put(constraint.getRow(), constraint); }
        public void addConstraints(Collection<Position> constraints) {
            for (Position constraint: constraints) { this.constraints.put(constraint.getRow(), constraint); }
        }
        public Collection<Position> getConstraints() { return this.constraints.values(); }
        public Collection<Integer> getPiorities() { return this.constraints.keySet(); }
        public HashMap<Integer, Position> getMappedConstraints() { return this.constraints; }
        public int getSenderId() { return this.senderId; }
        public boolean isActive(HashMap<Integer, Position> agentView, int id) {
	        for (Integer j: constraints.keySet()) {
	            if (j == id) continue;
	            if (!agentView.containsKey(j)) return false;
	            if (!agentView.get(j).equals(constraints.get(j))) return false;
            }
            return true;
        }
    }
    private static final String HANDLE_ADD_NEIGHBOR = "handleAddNeighbor";
	private static final String HANDLE_OK = "handleOk?";
    private static final String HANDLE_NO_GOOD = "handleNoGood";
    private static final String HANDLE_CONSISTENT = "handleConsistent";
    private static final String HANDLE_UPDATED = "handleUpdated";
    private static final String HANDLE_FINISH = "handleFinish";
	private static final Pattern pattern = Pattern.compile("StringContent \\[content=(.*?)\\]");
    private HashMap<Integer, Position> localView = new HashMap<>();
    private HashMap<Position, HashMap<Integer, Position>> finalView = new HashMap<>();
    private HashMap<Position, Integer> goods = new HashMap<>();
    private ArrayList<NoGood> constraints = new ArrayList<>();
	private HashSet<Integer> neighbors = new HashSet<>();
	private Position xi;
	private int id;
	private int n;
	private boolean finished = false;
	private Random rand = new Random();

	private void print() {
		for (int i=0; i < n; i++) {
			System.out.print("|");
			for (int j=0; j < n; j++) {
				if (id == i) {
					if (j == xi.getCol()) {
						System.out.print((char)27 + "[31m" + id + (char) 27 + "[30m|");
					} else System.out.print(" |");
				} else {
					if (localView.containsKey(i) && localView.get(i).getCol() == j && isHigherPriority(i)) {
						System.out.print(i + "|");
					} else System.out.print(" |");
				}
			}
			System.out.println();
		}
	}

	public boolean handleOk(int j, Position xj) {
        localView.put(j, xj);
        neighbors.add(j);
		return checkLocalView();
	}

	public boolean isHigherPriority(int j) { return j < id; }

	public boolean isConsistent() {
		for (Integer key: localView.keySet()) {
            if (id == key || !isHigherPriority(key)) continue;

			// Checks for row encounter
			if (localView.get(key).matchingRows(xi)) return false;

			// Checks for column encounter
			if (localView.get(key).matchingColumns(xi)) return false;

			// Checks for diagonal encounter
			if (localView.get(key).matchingDiagonals(xi)) return false;
		}

		// Checks if current position is in conflict with noGoods
        for (NoGood constraint: constraints) {
            for (Position banned: constraint.getConstraints()) {
                if (!constraint.isActive(localView, id)) continue;
                if (banned.equals(xi)) return false;
            }
        }

		return true;
	}

	public void filterWithNoGoods(ArrayList<Position> positions) {
	    nextValidPosition:
        for (Iterator<Position> i = positions.iterator(); i.hasNext();) {
            Position currentPosition = i.next();
            for (NoGood constraint: constraints) {
                if (!constraint.isActive(localView, id)) continue;
                for (Position banned: constraint.getConstraints()) {
                    if (banned.equals(currentPosition)) {
                        i.remove();
                        continue nextValidPosition;
                    }
                }
            }
        }
    }

    public void filterWithNoGoods(HashSet<Position> positions) {
        nextValidPosition:
        for (Iterator<Position> i = positions.iterator(); i.hasNext();) {
            Position currentPosition = i.next();
            for (NoGood constraint: constraints) {
                if (!constraint.isActive(localView, id)) continue;
                for (Position banned: constraint.getConstraints()) {
                    if (banned.equals(currentPosition)) {
                        i.remove();
                        continue nextValidPosition;
                    }
                }
            }
        }
    }

	public ArrayList<Position> consistentValuesInDomain() {
		ArrayList<Position> validPositions = new ArrayList<>();
		nextPosition:
		for (int j = 0; j < n; j++) {
			Position currentPosition = new Position(id, j);
			for (Integer key: localView.keySet()) {
                if (id == key || !isHigherPriority(key)) continue;

                if (localView.get(key).matchingRows(currentPosition) ||
						localView.get(key).matchingColumns(currentPosition) ||
						localView.get(key).matchingDiagonals(currentPosition)) continue nextPosition;
			}
			validPositions.add(currentPosition);
		}

		filterWithNoGoods(validPositions);
		return validPositions;
	}

	public void sendHandleOkToNeighbors() {
	    for (Integer j = id + 1; j < n; j++) {
	        if (id == j || isHigherPriority(j)) continue;
	        goods.put(xi, 0);
	        sendMessage(j.toString(), new MapContent(HANDLE_OK, new Content(xi)));
	    }
	}

	public NoGood generateNoGood() {
	    HashSet<Position> invalidPositions = new HashSet<>();
        HashMap<Position, HashMap<Integer, Position>> noGoods = new HashMap<>();
        for (int col = 0; col < n; col++) {
            Position currentPosition = new Position(id, col);
            for (Integer j: localView.keySet()) {
                if (id == j || !isHigherPriority(j)) continue;

                if (localView.get(j).matchingRows(currentPosition) ||
                        localView.get(j).matchingColumns(currentPosition) ||
                        localView.get(j).matchingDiagonals(currentPosition)) {
                    if (!noGoods.containsKey(currentPosition)) noGoods.put(currentPosition, new HashMap<Integer, Position>());
                    noGoods.get(currentPosition).put(j, localView.get(j));
                    invalidPositions.add(currentPosition);
                }
            }
        }
        filterWithNoGoods(invalidPositions);
        NoGood constraint = new NoGood(id);
        for (Position pos: invalidPositions) {
            for (Position conflicting: noGoods.get(pos).values()) { constraint.addConstraint(conflicting); }
        }

        return constraint;
	}

	public void backtrack() {
        NoGood constraint = generateNoGood();
		if (constraint.getConstraints().size() == 0) {
		    notifySolutionDoesNotExist();
		    return;
        }
        Integer minPriority = Collections.max(constraint.getPiorities());

        sendMessage(minPriority.toString(), new MapContent(HANDLE_NO_GOOD, new Content(constraint)));
//        localView.remove(minPriority);
    }

	public boolean checkLocalView() {
		if (!isConsistent()) {
			ArrayList<Position> validPositions = consistentValuesInDomain();
			if (validPositions.isEmpty()) {
			    backtrack();
                return false;
            } else {
                xi = validPositions.get(rand.nextInt(validPositions.size()));
                sendHandleOkToNeighbors();
                return true;
            }
		} else return true;
	}

	public void handleAddNeighbor(Integer j) {
	    neighbors.add(j);
	}

	public void handleNoGood(Integer j, NoGood constraint) {
	    constraints.add(constraint);

        for (Integer k: constraint.getPiorities()) {
            if (id != k && !localView.containsKey(k)) {
                sendMessage(k.toString(), new StringContent(HANDLE_ADD_NEIGHBOR));
                neighbors.add(k);
                localView.put(k, constraint.getMappedConstraints().get(k));
            }
        }

        Position oldValue = xi;
		checkLocalView();
		if (!oldValue.equals(xi)) {
            goods.put(xi, 0);
            sendMessage(j.toString(), new MapContent(HANDLE_OK, new Content(xi)));
        }
    }

    public boolean isLowestPriority() { return id == n - 1; }

    public boolean isHighestPriority() { return id == 0; }

    public QueenAgent(int agentId, int nAgents) {
		// Leave this method as it is...
    	super(agentId, nAgents);
	}

	@Override
	protected void start(int agentId, int nAgents) {
		// This method is called when the agent is initialized.
		// e.g., you can start sending messages:
        n = nAgents;
        id = agentId;
        xi = new Position(id, 0);
        goods.put(xi, 0);
        sendHandleOkToNeighbors();
	}

	@Override
	protected void processMessages(List<Message> newMessages) {
		// This method is called whenever there are any new messages for the robot

		// You can list the messages like this:
        for (Message message : newMessages) {
			Content wrappingContent = message.getContent();
			Matcher m = pattern.matcher(wrappingContent.toString());
			Content content;
			int j;
            Position xj;
			NoGood constraint;
			if (m.find()) {
				switch (m.group(1)) {
					case HANDLE_ADD_NEIGHBOR:
						j = Integer.parseInt(message.getSender());
						handleAddNeighbor(j);
						break;
					case HANDLE_OK:
						j = Integer.parseInt(message.getSender());
                        content = (Content) wrappingContent.getData();
						xj = (Position) content.getData();
						if (handleOk(j, xj) && isLowestPriority()) {
						    sendMessage(Integer.toString(id - 1), new StringContent(HANDLE_CONSISTENT));
                        }
						break;
					case HANDLE_NO_GOOD:
						j = Integer.parseInt(message.getSender());
						content = (Content) wrappingContent.getData();
						constraint = (NoGood) content.getData();
						handleNoGood(j, constraint);
						break;
                    case HANDLE_CONSISTENT:
                        if (isConsistent() && isHighestPriority()) sendMessage(Integer.toString(id + 1), new MapContent(HANDLE_UPDATED, new Content(xi)));
                        else if (isConsistent()) sendMessage(Integer.toString(id - 1), new StringContent(HANDLE_CONSISTENT));
                        break;
                    case HANDLE_UPDATED:
                        j = Integer.parseInt(message.getSender());
                        content = (Content) wrappingContent.getData();
                        xj = (Position) content.getData();
                        if (localView.containsKey(j) && localView.get(j).equals(xj) && isConsistent()) {
                            if (isLowestPriority()) {
                                if (finished) break;
								finished = true;
                                notifySolutionFound(xi.getCol());
                                broadcast(new StringContent(HANDLE_FINISH));
                            } else sendMessage(Integer.toString(id + 1), new MapContent(HANDLE_UPDATED, new Content(xi)));
                        }
                        break;
                    case HANDLE_FINISH:
                        if (finished) break;
                        finished = true;
                        notifySolutionFound(xi.getCol());
                        break;
                    default:
                }
//                System.out.println(getAgentId() + ": Received a message from " + message.getSender() + " with the content " + wrappingContent.toString());
            }

//            try {
//			    print();
//
//                Thread.sleep(100);
//            } catch (InterruptedException e) {}

        }
	}
}