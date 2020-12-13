package student;

import mas.agents.Message;
import mas.agents.task.mining.StatusMessage;

public class RequestPickup extends Message {
    public int sender;

    public RequestPickup(StatusMessage message, int agent) {
        sender = agent;
    }
}
