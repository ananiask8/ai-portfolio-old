package wumpus.agent;
import wumpus.world.Action;
import wumpus.world.Outcome;
import wumpus.world.WorldModel;
import wumpus.world.WorldState;

import javax.vecmath.Point2i;
import java.util.HashMap;
import java.util.Random;


public class Node {
    private WorldState state;
    private Action a;
    private Node parent;
    private boolean isExpanded;
    private HashMap<Point2i, Node> children = new HashMap<>();
    public HashMap<Action, Double> Q = new HashMap<>();
    private int visited = 0;
    public HashMap<Action, Integer> n = new HashMap<>();
    private Node lastNodeSelected;
    private Action lastActionSelected;
    private double r;

    public Node(WorldState state, Action a, Node parent, double r){
        this.parent = parent;
        this.state = state;
        this.r = r;
        this.a = a;
        this.isExpanded = false;
        this.Q = new HashMap<>();
        for (Action action: WorldModel.getActions(state)) { this.Q.put(action, (double) 0); }
        for (Action action: WorldModel.getActions(state)) { this.n.put(action, 0); }
    }

    public Node getParent() { return parent; }

    public double getReward() { return r; }

    public void setReward(double r) { this.r = r; }

    public HashMap<Point2i, Node> getChildren() { return children; }

    private Node getChild(WorldState s, Action a, double r) {
        Point2i p = new Point2i(s.getX(), s.getY());
        if (!children.containsKey(p)) children.put(p, new Node(s, a, this, r));
        return children.get(p);
    }

    public WorldState getState() { return state; }

    public void setAction(Action a) { this.a = a; }

    public Action getAction() { return a; }

    public boolean isExpanded() { return isExpanded; }

    public Node expand() {
        Action untriedAction = null;
        for (Action a: WorldModel.getActions(state)) {
            if(n.get(a) == 0) {
                untriedAction = a;
                break;
            }
        }

        Outcome o = WorldModel.performAction(state, untriedAction, new Random());
        lastNodeSelected = getChild(o.state, untriedAction, getReward() + o.reward);
        lastActionSelected = untriedAction;
        lastNodeSelected.setAction(untriedAction);
        return lastNodeSelected;
    }

    public Node getBestChild(int c) {
        Node v = null;
        Action selected = null;
        double maxValue = Integer.MIN_VALUE;
        for (Action a: WorldModel.getActions(state)) {
            Outcome o = WorldModel.performAction(state, a, new Random());
            WorldState s = o.state;
            double r = o.reward;

            double UCT = (Q.get(a)/n.get(a)) + c*Math.sqrt((2 * Math.log(visited) / n.get(a)));
            if (c == 0) System.out.println(a + " " + UCT + " : " + Q.get(a) + " " + n.get(a));
            if (maxValue < UCT) {
                maxValue = UCT;
                selected = a;
                // Child can be created even if action is not untried, due to probabilistic actions
                v = getChild(s, a, getReward() + r);
            }
        }
        lastNodeSelected = v;
        lastActionSelected = selected;
        v.setAction(selected);
        return v;
    }
    public Node getLastNodeSelected() { return lastNodeSelected; }
    public Action getLastActionSelected() { return lastActionSelected; }
    public void update(double r) {
        Action a = getLastActionSelected();
        visited++;
        n.put(a, n.get(a) + 1);
        Q.put(a, Q.get(a) + r);
        if (!isExpanded) {
            boolean allOpen = true;
            for (Action action: WorldModel.getActions(state)) allOpen &= n.get(action) > 0;
            isExpanded = allOpen;
        }
    }
}
