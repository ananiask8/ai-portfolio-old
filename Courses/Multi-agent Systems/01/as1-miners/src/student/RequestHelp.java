package student;

import mas.agents.Message;
import mas.agents.task.mining.StatusMessage;

public class RequestHelp extends Message {
    public int x;
    public int y;
    public int sender;

    public RequestHelp(StatusMessage message, int agent) {
        x = message.agentX;
        y = message.agentY;
        sender = agent;
    }
}
