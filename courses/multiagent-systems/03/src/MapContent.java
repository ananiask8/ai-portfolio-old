package mas.agent.student;

import cz.agents.alite.communication.content.Content;

@SuppressWarnings("serial")
public class MapContent extends Content {
    String key;
    Content content;

    public MapContent(String key, Content content) {
        super(content);
        this.key = key;
        this.content = content;
    }

    public String getKey() {
        return key;
    }

    @Override
    public String toString() {
        return "StringContent [content=" + key + "]";
    }

}
