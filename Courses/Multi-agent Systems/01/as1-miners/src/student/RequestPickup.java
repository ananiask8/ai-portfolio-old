package student;

import mas.agents.Message;
import mas.agents.task.mining.StatusMessage;

public class RequestPickup extends Message {
    public int x;
    public int y;
    public int sender;

    public RequestPickup(StatusMessage message, int agent) {
        x = message.agentX;
        y = message.agentY;
        sender = agent;
    }
}