package student;

import mas.agents.Message;
import mas.agents.task.mining.StatusMessage;

public class RequestHelp extends Message {
    public int x;
    public int y;
    public int sender;

    public RequestHelp(int[] pos, int agent) {
        x = pos[0];
        y = pos[1];
        sender = agent;
    }

    public RequestHelp(StatusMessage m, int agent) {
        x = m.agentX;
        y = m.agentY;
        sender = agent;
    }
}
