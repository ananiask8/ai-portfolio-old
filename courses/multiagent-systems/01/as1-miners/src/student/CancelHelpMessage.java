package student;

import mas.agents.Message;
import mas.agents.task.mining.StatusMessage;

public class CancelHelpMessage extends Message {
    public int x;
    public int y;
    public int sender;

    public CancelHelpMessage (int[] pos, int agent) {
        x = pos[0];
        y = pos[1];
        sender = agent;
    }
}
