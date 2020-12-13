package student;

import mas.agents.Message;

public class MapUpdate extends Message {
    public int x;
    public int y;
    public int type;
    public int sender;

    public MapUpdate(int[] pos, int type, int agent) {
        x = pos[0];
        y = pos[1];
        this.type = type;
        sender = agent;
    }
}
