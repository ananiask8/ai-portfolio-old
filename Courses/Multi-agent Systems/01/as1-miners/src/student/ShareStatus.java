package student;
import mas.agents.Message;
import mas.agents.task.mining.StatusMessage;
import student.Agent;

import java.util.List;

public class ShareStatus extends Message {
    public int x;
    public int y;
    public List<StatusMessage.SensorData> sensorInput;
    public int[] pos;
    public int sender;

    public ShareStatus(StatusMessage message, int agent) {
        x = message.agentX;
        y = message.agentY;
        sensorInput = message.sensorInput;
        pos = new int[] {message.agentX, message.agentY};
        sender = agent;
    }
}
