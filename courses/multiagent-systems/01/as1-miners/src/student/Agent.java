package student;

import mas.agents.*;
import mas.agents.task.mining.*;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;

public class Agent extends AbstractAgent {
    private void addObjectAt(int x, int y, int type) {
        map[x][y] = type;
    }

    private void resetMap() {
        for(int i = 0; i < map.length; i++) {
            for(int j = 0; j < map[i].length; j++) {
                if (map[i][j] != 1 && map[i][j] != 2) map[i][j] = 0;
            }
        }
    }

    private void removeObjectAt(int x, int y) { map[x][y] = 5; }
    private boolean containsAt(int x, int y, int type) { return type == map[x][y]; }
    private boolean containsGoldAt(int[] pos) { return containsAt(pos[0], pos[1], 3); }
    private int[][] getLocationsOf(int type) {
        ArrayList<int[]> result = new ArrayList<>();
        for (int i = 0; i < map.length; i++) {
            for (int j = 0; j < map[i].length; j++) {
                if (map[i][j] == type) {
                    int[] pos = {i, j};
                    result.add(pos);
                }
            }
        }

        return result.toArray(new int[result.size()][]);
    }

    private int[][] getObstacleLocations() {
        return getLocationsOf(1);
    }

    private int[][] getVisitedLocations() {
        ArrayList<int[]> result = new ArrayList<>();
        for (int i = 0; i < map.length; i++) {
            for (int j = 0; j < map[i].length; j++) {
                if (map[i][j] != 0) {
                    int[] pos = {i, j};
                    result.add(pos);
                }
            }
        }

        return result.toArray(new int[result.size()][]);
    }


    private String getLocationsToString(int type) {
        String result = "";
        for (int[] location : getLocationsOf(type)) {
            result += "[" + location[0] + "," + location[1] + "]\r\n";
        }

        return result;
    }

    private int[] getClosestPositionOf(Integer[] types) throws IOException {
        int maxLayers = (int) Math.ceil((map.length - 1));
        StatusMessage status = sense();
        for (int layer = 1; layer < maxLayers; layer++) {
            int xmin = status.agentX - layer > 0 ? status.agentX - layer : 0;
            int xmax = status.agentX + layer < map.length - 1 ? status.agentX + layer : map.length - 1;
            int ymin = status.agentY - layer > 0 ? status.agentY - layer : 0;
            int ymax = status.agentY + layer < map.length - 1 ? status.agentY + layer : map[0].length - 1;

            for (int j = ymin; j <= ymax; j++) {
                for (int i = xmin; i <= xmax; i++) {
                    if (i == xmin || i == xmax || j == ymin || j == ymax) {
                        if (Arrays.asList(types).contains(map[i][j])) return new int[]{i, j};
                    }
                }
            }
        }

        return new int[] {};
    }

    private HashSet<Integer> getMovesFor(Integer[] types) throws IOException {
        StatusMessage status = sense();
        HashSet<Integer> options = new HashSet<>();
        if (status.agentY > 0 && Arrays.asList(types).contains(map[status.agentX][status.agentY - 1])) options.add(0);
        if (map[status.agentX].length > status.agentY + 1 && Arrays.asList(types).contains(map[status.agentX][status.agentY + 1])) options.add(2);
        if (map.length > status.agentX + 1 && Arrays.asList(types).contains(map[status.agentX + 1][status.agentY])) options.add(1);
        if (status.agentX > 0 && Arrays.asList(types).contains(map[status.agentX - 1][status.agentY])) options.add(3);

        return options;
    }

    private int[] getDepotPosition() {
        for(int i = 0; i < map.length; i++) {
            for(int j = 0; j < map[i].length; j++) {
                if (map[i][j] == 2) return new int[] {i, j};
            }
        }

        return new int[] {};
    }

    private void initializeLocations() {
        map = new int[currentStatus.width][currentStatus.height];
    }

    private void addGoldAt(int x, int y) {
        addObjectAt(x, y, 3);
    }

    private int[][] getGoldLocations() {
        return getLocationsOf(3);
    }

    private String goldLocationsToString() {
        return "GOLD:\r\n" + getLocationsToString(3);
    }

    private void addDepotAt(int x, int y) {
        hasDepot = true;
        addObjectAt(x, y, 2);
    }

    private void addObstacleAt(int x, int y) {
        addObjectAt(x, y, 1);
    }

    // See also class StatusMessage
    public static String[] types = {
            "", "obstacle", "depot", "gold", "agent", "visited"
    };

    public static String[] quadrants = {
            "upper-left", "lower-left", "upper-right", "lower-right"
    };

    private int[][] map;
    private ArrayList<int[]> friendLocations = new ArrayList<>(4);
    private StatusMessage currentStatus;
    private boolean hasGold = false;
    private boolean hasDepot = false;
    private boolean isWaitingForHelp = false;
    private boolean isWaitingForPickup = false;
    private boolean isHelping = false;
    private boolean isHelpOnTheWay = false;

    public Agent(int id, InputStream is, OutputStream os, SimulationApi api) throws IOException, InterruptedException {
        super(id, is, os, api);
        friendLocations = new ArrayList<>(Arrays.asList(new int[] {0, 0}, new int[] {0, 0}, new int[] {0, 0}, new int[] {0, 0}));
    }

    public int[] getClosestPositionUnvisited() throws IOException {
        Integer[] types = {0};
        return getClosestPositionOf(types);
    }

    public int[] getClosestPositionGold() throws IOException {
        Integer[] types = {3};
        return getClosestPositionOf(types);
    }


    public int[] getBestGold() throws IOException {
        int[][] goldLocations = getGoldLocations();

        int min = Integer.MAX_VALUE;
        int best = 0;
        for (int i = 0; i < goldLocations.length; i++) {
            for (int j = 0; j < friendLocations.size(); j++) {
                if (j != getAgentId() - 1) {
                    int dMe = Math.abs(currentStatus.agentX - goldLocations[i][0]) + Math.abs(currentStatus.agentY - goldLocations[i][1]);
                    int dFriend = Math.abs(friendLocations.get(j)[0] - goldLocations[i][0]) + Math.abs(friendLocations.get(j)[1] - goldLocations[i][1]);
                    int d = dMe + dFriend;
                    if (d < min) {
                        min = d;
                        best = i;
                    }
                }
            }
        }

        return goldLocations[best];
    }

    public void updateStatus() throws IOException {
        markAsVisited();
        friendLocations.set(getAgentId() - 1, new int[] {currentStatus.agentX, currentStatus.agentY});
        for(StatusMessage.SensorData data : currentStatus.sensorInput) {
            if (map[data.x][data.y] != data.type) {
                if (Objects.equals(types[data.type], "visited")) {
                    markAsVisited(new int[] {data.x, data.y});
                }
                if (Objects.equals(types[data.type], "gold")) addGoldAt(data.x, data.y);
                if (Objects.equals(types[data.type], "depot")) addDepotAt(data.x, data.y);
                if (Objects.equals(types[data.type], "obstacle")) addObstacleAt(data.x, data.y);
            }
        }
    }

    public void storeStatus(Message m) throws Exception {
        ShareStatus status = (ShareStatus) m;
        markAsVisited(status.pos);
        friendLocations.set(status.sender - 1, status.pos);
        for(StatusMessage.SensorData data : status.sensorInput) {
            if (map[data.x][data.y] != data.type) {
                if (Objects.equals(types[data.type], "gold")) addGoldAt(data.x, data.y);
                if (Objects.equals(types[data.type], "depot")) addDepotAt(data.x, data.y);
                if (Objects.equals(types[data.type], "obstacle")) addObstacleAt(data.x, data.y);
            }
        }
    }

    public void shareStatus() throws IOException {
        for(int i = 1; i <= 4; i++) {
            if (i != getAgentId()) sendMessage(i, new ShareStatus(currentStatus, getAgentId()));
        }
    }

    public void sendMapUpdate(int[] pos, int type) throws IOException {
        for(int i = 1; i <= 4; i++) {
            sendMessage(i, new MapUpdate(pos, type, getAgentId()));
        }
    }

    public void updateMap(Message m) {
        MapUpdate update = (MapUpdate) m;
        map[update.x][update.y] = update.type;
    }

    public Integer[] getAllAvailableMoves() throws IOException {
        return getAvailableMovesFrom(new HashSet<>(Arrays.asList(0, 1, 2, 3)));
    }

    public Integer[] getAllMoves() {
        return new Integer[] {0, 1, 2, 3};
    }

    public Integer[] getAvailableMovesFrom(HashSet<Integer> options) throws IOException {
        StatusMessage status = sense();

        if (status.agentY == 0) options.remove(0);
        if (status.agentX == map.length - 1) options.remove(1);
        if (status.agentY == map[status.agentX].length - 1) options.remove(2);
        if (status.agentX == 0) options.remove(3);

        for(StatusMessage.SensorData data : status.sensorInput) {
            if (Objects.equals(types[data.type], "obstacle") || Objects.equals(types[data.type], "agent")) {
                if (data.x - status.agentX == 0 && data.y - status.agentY == -1) options.remove(0);
                if (data.x - status.agentX == 1 && data.y - status.agentY == 0) options.remove(1);
                if (data.x - status.agentX == 0 && data.y - status.agentY == 1) options.remove(2);
                if (data.x - status.agentX == -1 && data.y - status.agentY == 0) options.remove(3);
            }
        }

        ArrayList<Integer> moves = new ArrayList<>();
        for(Integer i : options) {
            moves.add(i);
        }

        return moves.toArray(new Integer[moves.size()]);
    }

    public HashSet<Integer> getUnvisited() throws IOException {
        Integer[] priority = {0};
        HashSet<Integer> result = getMovesFor(priority);

        return result;
    }

    public HashSet<Integer> getGold() throws IOException {
        Integer[] priority = {3};
        HashSet<Integer> result = getMovesFor(priority);

        return result;
    }

    public void performRandomSequence(int times) throws IOException {
        int prev = -1;
        for (int i = 0; i < times; i++) {
            Random rand = new Random();
            HashSet<Integer> moveSet = new HashSet<>(Arrays.asList(0, 1, 2, 3));
            moveSet.remove(prev);
            Integer[] availableMoves = getAvailableMovesFrom(moveSet);
            if (availableMoves.length == 0) break;
            int moveIndex = rand.nextInt(availableMoves.length);
            switch (availableMoves[moveIndex]) {
                case 0:
                    currentStatus = up();
                    prev = 2;
                    break;
                case 1:
                    currentStatus = right();
                    prev = 3;
                    break;
                case 2:
                    currentStatus = down();
                    prev = 0;
                    break;
                default:
                    currentStatus = left();
                    prev = 1;
                    break;
            }
            updateStatus();
            shareStatus();
        }
    }

    public void performRandomMove(Integer[] availableMoves) throws IOException {
        if (availableMoves.length <= 0) {
            currentStatus = sense();
            return;
        }

        Random rand = new Random();
        int moveIndex = rand.nextInt(availableMoves.length);
        switch (availableMoves[moveIndex]) {
            case 0: currentStatus = up();
                break;
            case 1: currentStatus = right();
                break;
            case 2: currentStatus = down();
                break;
            default: currentStatus = left();
                break;
        }
        updateStatus();
        shareStatus();
    }

    public void markAsVisited() throws IOException {
        StatusMessage status = sense();
        if (map[status.agentX][status.agentY] == 0) {
            map[status.agentX][status.agentY] = 5;
        }
    }

    public void markAsVisited(int[] pos) {
        if (map[pos[0]][pos[1]] == 0) {
            map[pos[0]][pos[1]] = 5;
        }
    }

    public void mapToString() {
        String output = "";
        for (int j = 0; j < map[0].length; j++) {
            for (int i = 0; i < map.length; i++) {
                output += map[i][j] + " |";
            }
            output += "\r\n";
        }
        output += "\r\n\r\n\r\n";
        System.out.println(output);
    }

    public boolean allLocationsAreVisited() {
        boolean result = true;
        for(int i = 0; i < map.length; i++) {
            for(int j = 0; j < map[i].length; j++) {
                result &= map[i][j] != 0;
            }
        }

        return result;
    }

    public boolean shouldExploreMore() {
        // Explore up to the 50% of the map
        return getVisitedLocations().length < 0.7 * (map.length * map[0].length);
    }

    public boolean mapIsOfNarrowSetting() {
        return (double) getObstacleLocations().length / getVisitedLocations().length > 0.3;
    }

    public boolean isFriendNearby() throws IOException {
        if (friendLocations.isEmpty()) return false;

        for (int id = 0; id < friendLocations.size(); id++) {
            int[] friend = friendLocations.get(id);
            if ((getAgentId() != id + 1) && Math.abs(friend[0] - currentStatus.agentX) + Math.abs(friend[1] - currentStatus.agentY) < 20 ) return true;
        }

        return false;
    }

    public boolean isFriendInPosition(int[] pos) {
        for (int[] friend: friendLocations) {
            if (friend[0] == pos[0] && friend[1] == pos[1]) return true;
        }

        return false;
    }


    public boolean isFriendInPosition(int[] pos, int id) {
        return friendLocations.get(id)[0] == pos[0] && friendLocations.get(id)[1] == pos[1];
    }

    public int getFriendInPosition(int[] pos) {
        for (int i = 0; i < friendLocations.size(); i++) {
            if (isFriendInPosition(pos, i)) return i;
        }

        return -1;
    }

    public int getClosestFriend() {
        int min = Integer.MAX_VALUE;
        int result = -1;
        for (int id = 0; id < friendLocations.size(); id++) {
            int[] friend = friendLocations.get(id);
            Integer[] steps = BFS(friend, new Integer[] {0, 2, 3, 4, 5}, false);
            if ((getAgentId() != id + 1) && steps.length < min) {
                min = steps.length;
                result = id;
            }
        }

        return result + 1;
    }

    public boolean isGoldNearby() throws IOException {
        if (getGoldLocations().length == 0) return false;

        for (int[] pos: getGoldLocations()) {
            if (Math.abs(pos[0] - currentStatus.agentX) + Math.abs(pos[1] - currentStatus.agentY) < 16 ) return true;
        }

        return false;
    }

    public ArrayList<int[]> getSuccessors(int[] pos, Integer[] allowedTypes, boolean conditions) {
        ArrayList<int[]> result = new ArrayList<>();
        List<Integer> types = Arrays.asList(allowedTypes);

        int[] up = new int[] {pos[0], pos[1] - 1, 0};
        if (pos[1] - 1 >= 0 && types.contains(map[pos[0]][pos[1] - 1]) && (!conditions || !isFriendInPosition(up) ||
                (isFriendInPosition(up) && containsGoldAt(up) && isHelping))) result.add(up);

        int[] right = new int[] {pos[0] + 1, pos[1], 1};
        if (pos[0] + 1 < map.length && types.contains(map[pos[0] + 1][pos[1]]) && (!conditions || !isFriendInPosition(right) ||
                (isFriendInPosition(right) && containsGoldAt(right) && isHelping))) result.add(right);

        int[] down = new int[] {pos[0], pos[1] + 1, 2};
        if (pos[1] + 1 < map[0].length && types.contains(map[pos[0]][pos[1] + 1]) && (!conditions || !isFriendInPosition(down) ||
                (isFriendInPosition(down) && containsGoldAt(down) && isHelping))) result.add(down);

        int[] left = new int[] {pos[0] - 1, pos[1], 3};
        if (pos[0] - 1 >= 0 && types.contains(map[pos[0] - 1][pos[1]]) && (!conditions || !isFriendInPosition(left) ||
                (isFriendInPosition(left) && containsGoldAt(left) && isHelping))) result.add(left);

        return result;
    }

    public Integer[] buildPath(int[] state, HashMap<String, int[]> steps) {
        ArrayDeque<Integer> path = new ArrayDeque<>();

        String stateCode = Arrays.toString(state);
        int action = steps.get(stateCode)[2];
        while (action != -1) {
            path.push(action);

            int[] step = steps.get(stateCode);
            state = new int[] {step[0], step[1]};
            stateCode = Arrays.toString(state);
            action = steps.get(stateCode)[2];
        }

        return path.toArray(new Integer[path.size()]);
    }

    public boolean openContains(LinkedList<int[]> open, int[] contains) {
        for (int[] pos: open) {
            boolean contained = true;
            for (int i = 0; i < contains.length; i++) {
                if (i < pos.length) contained &= contains[i] == pos[i];
                else contained = false;
            }
            if (contained) return true;
        }

        return false;
    }

    public Integer[] BFS(int[] goal, Integer[] types, boolean conditions) {
        LinkedList<int[]> open = new LinkedList<>();
        HashSet<String> closed = new HashSet<>();
        int[] parent;
        HashMap<String, int[]> meta = new HashMap<>();

        int[] startAndAction = new int[] {currentStatus.agentX, currentStatus.agentY, -1};
        int[] start = new int[] {startAndAction[0], startAndAction[1]};
        meta.put(Arrays.toString(start), startAndAction);
        open.addLast(start);

        while (!open.isEmpty()) {
            parent = open.removeFirst();
            if (goal[0] == parent[0] && goal[1] == parent[1]) return buildPath(parent, meta);
            for (int[] childAndAction: getSuccessors(parent, types, conditions)) {
                int[] child = new int[] {childAndAction[0], childAndAction[1]};
                int action = childAndAction[2];

                int[] parentAndAction = new int[] {parent[0], parent[1], action};
                if (!closed.contains(Arrays.toString(child)) && !openContains(open, child)) {
                    meta.put(Arrays.toString(child), parentAndAction);
                    open.addLast(child);
                }
            }
            closed.add(Arrays.toString(parent));
        }

        return new Integer[] {};
    }

    public void walk(int step) throws IOException {
        switch (step) {
            case 0: currentStatus = up();
                break;
            case 1: currentStatus = right();
                break;
            case 2: currentStatus = down();
                break;
            case 3: currentStatus = left();
                break;
        }
    }

    public boolean isAgentInPos(int[] pos) {
        return currentStatus.agentX == pos[0] && currentStatus.agentY == pos[1];
    }

    public int walkPath(Integer[] steps) throws Exception {
        for (int i = 0; i < steps.length; i++) {
            int[] previous = new int[] {currentStatus.agentX, currentStatus.agentY};
            if (!satisfyRequests() || !isAgentInPos(previous)) return -1;
            walk(steps[i]);
            updateStatus();
            shareStatus();
            if (currentStatus.agentX == previous[0] && currentStatus.agentY == previous[1]) return i;
        }

        return steps.length;
    }

    public boolean goTo(int[] pos) throws Exception { // when committing like this, not exploring / no side effects
        if (pos.length <= 0) return false;
        while (true) {
            if (!satisfyRequests()) return false;
            updateStatus();
            shareStatus();
            if (isAgentInPos(pos) || (isHelping &&
                    ((currentStatus.agentX - pos[0] == 0 && Math.abs(currentStatus.agentY - pos[1]) <= 1) ||
                    (Math.abs(currentStatus.agentX - pos[0]) <= 1 && currentStatus.agentY - pos[1] == 0)))) return true;
            Integer[] steps = BFS(pos, new Integer[] {0, 2, 3, 4, 5}, true);

            if (steps.length == 0) return false;

            int stepsTaken = walkPath(steps);
            if (stepsTaken == -1) return false;
        }
    }

    public void awaitForHelp() throws Exception {
        isWaitingForHelp = true;
        int trials = 0;
        while (isWaitingForHelp && trials < 20) {
            satisfyRequests();
            if (!isHelpOnTheWay) {
                sendMessage(getClosestFriend(), new RequestHelp (sense(), getAgentId()));
            }
            try {
                Thread.sleep(10);
            } catch(InterruptedException ie) {}
            trials++;
        }
        isWaitingForHelp = false;
        isHelpOnTheWay = false;
        performRandomSequence(5);
    }

    public void awaitPickup(int sender) throws Exception {
        isWaitingForPickup = true;
        sendMessage(sender, new RequestPickup (sense(), getAgentId()));
        while (isWaitingForPickup) {
            satisfyRequests();
        }
    }

    public void goToHelp(Message m) throws Exception {
        RequestHelp rh = (RequestHelp) m;
        sendMessage(rh.sender, new HelpOnTheWayMessage (sense(), getAgentId()));
        int[] pos = new int[] {rh.x, rh.y};
        log(String.format("Help at %d, %d", rh.x, rh.y));
        isHelping = true;
        int tries = 0;
        while (tries < 10) {
            if (goTo(pos)) {
                isHelping = false;
                awaitPickup(rh.sender);
                return;
            } else {
                performRandomSequence(5);
                tries++;
            }
        }
        isHelping = false;
        sendMessage(rh.sender, new CancelHelpMessage(new int[] {currentStatus.agentX, currentStatus.agentY}, getAgentId()));
        return;
    }

    public void goToDepot() throws Exception {
        int[] depot = getDepotPosition();
        log(String.format("Going to depot at %d, %d", depot[0], depot[1]));
        boolean isGoingToDepot = true;
        int tries = 0;
        while (tries < 20) {
            if (goTo(depot) && currentStatus.isAtDepot()) {
                hasGold = false;
                drop();
                break;
            } else {
                performRandomSequence(5);
                tries++;
            }
        }
        isGoingToDepot = false;
    }

    public void goToGold() throws Exception {
//        int[] gold = getClosestPositionGold();
        int[] gold = getBestGold();
        if (goTo(gold)) {
            if (currentStatus.isAtGold()) awaitForHelp();
            else sendMapUpdate(new int[]{currentStatus.agentX, currentStatus.agentY}, 5);
        } else performRandomSequence(5);
    }

    public boolean canPick() throws IOException {
        currentStatus = sense();
        for(StatusMessage.SensorData data : currentStatus.sensorInput) {
            if (map[data.x][data.y] != data.type) {
                if (Objects.equals(types[data.type], "agent")) {
                    if (Math.abs(data.x - currentStatus.agentX) + Math.abs(data.y - currentStatus.agentY) == 1) return true;
                }
            }
        }

        return false;
    }

    public boolean satisfyRequests() throws Exception {
        while (messageAvailable()) {
            Message m = readMessage();
            if (m instanceof ShareStatus) storeStatus(m);
            if (m instanceof MapUpdate) updateMap(m);
            if (isWaitingForHelp) {
                if (m instanceof HelpOnTheWayMessage) isHelpOnTheWay = true;
                if (m instanceof CancelHelpMessage) isHelpOnTheWay = false;
                // ---------------------------------------------------------------------------------------------------
                // Desire #3: Pick up the gold and drop it in the depot
                // ---------------------------------------------------------------------------------------------------
                if (m instanceof RequestPickup) {
                    while (!canPick());
                    RequestPickup rp = (RequestPickup) m;
                    if (isAgentInPos(new int[] {currentStatus.agentX, currentStatus.agentY + 1}) || isAgentInPos(new int[] {currentStatus.agentX, currentStatus.agentY - 1}) ||
                            isAgentInPos(new int[] {currentStatus.agentX + 1, currentStatus.agentY}) || isAgentInPos(new int[] {currentStatus.agentX - 1, currentStatus.agentY})); {
                        currentStatus = pick();
                        sendMessage(rp.sender, new ConfirmPickup(currentStatus));
                        hasGold = true;
                        isWaitingForHelp = false;
                        if (!currentStatus.isAtGold()) sendMapUpdate(new int[] {currentStatus.agentX, currentStatus.agentY}, 5);
                    }
                }
            }
            if (!isWaitingForHelp || hasGold) {
                if (m instanceof RequestPickup) {
                    RequestPickup rp = (RequestPickup) m;
                    sendMessage(rp.sender, new ConfirmPickup(currentStatus));
                }
            }
            if (isHelping) {
                if (m instanceof CancelHelpMessage) isHelping = false;
            }
            if (isWaitingForPickup) {
                if (m instanceof ConfirmPickup) isWaitingForPickup = false;
            }
            if (!isHelping && !isWaitingForPickup) {
                // ---------------------------------------------------------------------------------------------------
                // Desire #2: Go to help
                // ---------------------------------------------------------------------------------------------------
                if (m instanceof RequestHelp) {
                    isWaitingForHelp = false;
                    goToHelp(m);
                }
            }
        }
        return true;
    }

    public void performAction() throws Exception {
        HashSet<Integer> desirableMoves;
        Integer[] possibleMoves;

        // ---------------------------------------------------------------------------------------------------
        // Desire #5: Reset knowledge on end, to better search in case of new mines
        // ---------------------------------------------------------------------------------------------------
        if (allLocationsAreVisited() && getGoldLocations().length == 0) resetMap();

        // ---------------------------------------------------------------------------------------------------
        // Desire #3: Drop gold in the depot
        // ---------------------------------------------------------------------------------------------------
        // Beliefs:
        // 1 - Exploring the map is not important at all
        //     We know where the gold is
        //
        //     ACTION: Carry gold around until can go to depot
        // OR
        // 2 - Exploring the map is not important ANYMORE
        //     We have a good understanding of where the gold is
        //     We know where the depot is
        //
        //     ACTION: Take it to the depot with your friend
        // ---------------------------------------------------------------------------------------------------
        if (hasDepot && hasGold) {
            goToDepot();
            return;
        }

        // ---------------------------------------------------------------------------------------------------
        // Desire #1: Pick up gold
        // ---------------------------------------------------------------------------------------------------
        // Beliefs:
        // Know some gold locations
        //
        // ACTION: Move to gold, request help and wait for it
        // ---------------------------------------------------------------------------------------------------
        if (!hasGold && !isHelping && !isWaitingForHelp && getGoldLocations().length > 0) {
            goToGold();
            return;
        }

        // ---------------------------------------------------------------------------------------------------
        // Desire #0: Explore
        // ---------------------------------------------------------------------------------------------------
        // Beliefs:
        // There is a position, in immediate proximity, that I can explore
        //
        // ACTION: Move to unvisited position
        //
        // OR
        //
        // Beliefs:
        // 1 - The map is of open characteristics (not a lot of narrow paths)
        //   AND
        //     1 - Less than 50% (~ some threshold percentage) of the map has been explored
        //       OR
        //     2 - The threshold has been met, but still don't know where the depot is
        // 2 - The map is of narrow characteristics
        //     Don't know where the gold is
        //
        // ACTION: Make a move that will get you closer to that position
        //         If not possible, perform a random move
        //
        // ---------------------------------------------------------------------------------------------------
        if ((!hasDepot || shouldExploreMore()) && !isWaitingForHelp && !isHelping) {
            desirableMoves = getUnvisited();
            possibleMoves = getAvailableMovesFrom(desirableMoves);
            if (possibleMoves.length > 0) {
                performRandomMove(possibleMoves);
                return;
            }
        }

        //---------------------------------------------------------------------------------------------------
        // Desire #6: ♩ ♪ ♫ ♬ ... And nothing else matters ... ♩ ♪ ♫ ♬
        // ---------------------------------------------------------------------------------------------------
        // ACTION: Just move somewhere
        // ---------------------------------------------------------------------------------------------------
        int[] pos = getClosestPositionUnvisited();
        if (!goTo(pos)) performRandomSequence(5);
//        performRandomMove(getAllAvailableMoves());


    }

    @Override
    public void act() throws Exception {
        currentStatus = sense();
        initializeLocations();

        while(true) {
            updateStatus();
            shareStatus();
            satisfyRequests();
            performAction();
            currentStatus = sense();
        }
    }
}
