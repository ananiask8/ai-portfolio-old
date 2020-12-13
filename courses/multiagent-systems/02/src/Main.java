package hilarjos;

import ilog.concert.IloException;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Main {
    private static String BASE_PATH = "/Users/ananias/Documents/CTU/Multi-Agent Systems/Labs/git/02/dataset/";
    private static Pattern ALLOWED_CHARS = Pattern.compile("^[-EGD].{0,2}");
    private static Pattern SEQUENTIABLE_CHARS = Pattern.compile("^[EGSD].{0,2}");
    private static Map<String, String> AGENT_ACTIONS;
    private static Map<String, String> BANDIT_ACTIONS;
    private static Map<String, Integer> SCORES;
    static {
        Map<String, String> agent = new HashMap<>();
        agent.put("start", "S");
        agent.put("bandit", "E");
        agent.put("gold", "G");
        agent.put("goal", "D");
        AGENT_ACTIONS = Collections.unmodifiableMap(agent);

        Map<String, String> bandits = new HashMap<>();
        bandits.put("alarm", "⚠ Alarm ⚠");
        bandits.put("attack", "⚔ Attack ⚔");
        bandits.put("succeed", "Succeed");
        bandits.put("fail", "Fail");
        BANDIT_ACTIONS = Collections.unmodifiableMap(bandits);

        Map<String, Integer> scores = new HashMap<>();
        scores.put(AGENT_ACTIONS.get("gold"), 1);
        scores.put(AGENT_ACTIONS.get("goal"), 10);
        scores.put(BANDIT_ACTIONS.get("succeed"), 0);
        SCORES = Collections.unmodifiableMap(scores);
    }

    private int rows;
    private int cols;
    private String[][] map;
    private int nBandits;
    private int dangerousLocations;
    private double pSuccessfulAttack;
    private int[] start = new int[] {};
    private int[] end = new int[] {};
    private ArrayList<ArrayList<String>> agentActions = new ArrayList<>();
    private ArrayList<ArrayList<String>> banditsActions = new ArrayList<>();
    private ArrayList<HashMap<String, ArrayList<ArrayList<String>>>> alarmMappings = new ArrayList<>();
    private ArrayList<ArrayList<String>> banditsCompleteActions = new ArrayList<>();
    private ArrayList<ArrayList<String>> banditsNoNature = new ArrayList<>();
    private HashMap<String, String> completeToNoNature = new HashMap<>();
    private ArrayList<HashMap<String, HashSet<String>>> banditInformationSets = new ArrayList<>();
    private ArrayList<HashSet<String>> agentInformationSets = new ArrayList<>();
    private ArrayList<HashSet<Integer>> banditToAgentActionList = new ArrayList<>();
    private HashMap<String, HashSet<String>> agentNFCs = new HashMap<>();
    private HashMap<String, HashMap<String, HashSet<String>>> banditsNFCs = new HashMap<>();
    private HashMap<String, HashSet<String>> differentialNFCs = new HashMap<>();
    private HashMap<String, HashMap<String, ArrayList<String>>> agentUtilities = new HashMap<>();
    private HashMap<String, HashMap<String, ArrayList<String>>> banditUtilities = new HashMap<>();
    private IloCplex cplex;
    private HashMap<String, IloNumVar> cplexVars = new HashMap<>();
    private HashMap<String, HashMap<String, Double>> utilities = new HashMap<>();
    private HashSet<String> alarmTriggers = new HashSet<>();

    private int[] setStart() {
        int j = 0;
        for (String[] row: map) {
            int i = 0;
            for (String element: row) {
                if (element.equals("S")) return start = new int[] {j, i};
                i++;
            }
            j++;
        }

        return new int[] {};
    }

    private int[] getStart() {
        if (start.length > 0) return start;

        return setStart();
    }

    private int[] setEnd() {
        int j = 0;
        for (String[] row: map) {
            int i = 0;
            for (String element: row) {
                if (element.equals("D")) return end = new int[] {j, i};
                i++;
            }
            j++;
        }

        return new int[] {};
    }

    private int[] getEnd() {
        if (end.length > 0) return end;

        return setEnd();
    }

    public void readInputFromCLI() {
        Iterator<String> lines;
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String line = reader.readLine();
            readInputFromDataset(line);
//            StringTokenizer st = new StringTokenizer(line, " ");
//            rows = Integer.parseInt(st.nextToken());
//
//            line = reader.readLine();
//            st = new StringTokenizer(line, " ");
//            cols = Integer.parseInt(st.nextToken());
//            map = new String[rows][cols];
//            dangerousLocations = 0;
//            int goldCount = 0;
//            for (int y = 0; y < rows; y++) {
//                line = reader.readLine();
//                st = new StringTokenizer(line, " ");
//                for (int x = 0; x < cols; x++) {
//                    map[y][x] = Character.toString(line.charAt(x));
//                    if (map[y][x].equals("E")) map[y][x] += ++dangerousLocations;
//                    else if (map[y][x].equals("G")) map[y][x] += ++goldCount;
//                }
//            }
//
//            line = reader.readLine();
//            st = new StringTokenizer(line, " ");
//            nBandits = Integer.parseInt(st.nextToken());
//
//            line = reader.readLine();
//            st = new StringTokenizer(line, " ");
//            pSuccessfulAttack = Double.parseDouble(st.nextToken());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void readInputFromDataset(String path) {
        Iterator<String> lines;
        try {
            lines = Files.lines(Paths.get(path)).iterator();
            System.out.println();
            rows = Integer.parseInt(lines.next());
            cols = Integer.parseInt(lines.next());
            map = new String[rows][cols];
            dangerousLocations = 0;
            int goldCount = 0;
            for (int y = 0; y < rows; y++) {
                String row = lines.next();
                for (int x = 0; x < cols; x++) {
                    map[y][x] = Character.toString(row.charAt(x));
                    if (map[y][x].equals("E")) map[y][x] += ++dangerousLocations;
                    else if (map[y][x].equals("G")) map[y][x] += ++goldCount;
                }
            }
            nBandits = Integer.parseInt(lines.next());
            pSuccessfulAttack = Double.parseDouble(lines.next());
            System.out.println("\r\n{ Rows: " + rows + ", Cols: " + cols + ", Bandits: " + nBandits + ", P: " + pSuccessfulAttack + " }");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double readOutputFromDataset(String path) {
        Iterator<String> lines;
        try {
            lines = Files.lines(Paths.get(path)).iterator();
            String input = lines.next();
            return Double.parseDouble(input);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return 0;
    }

    public ArrayList<int[]> getSuccessors(int[] pos) {
        ArrayList<int[]> result = new ArrayList<>();

        int[] up = new int[] {pos[0] - 1, pos[1], 0};
        Matcher m = ALLOWED_CHARS.matcher(map[pos[0] - 1][pos[1]]);
        if (pos[0] - 1 >= 0 && m.find()) result.add(up);

        int[] right = new int[] {pos[0], pos[1] + 1, 1};
        m = ALLOWED_CHARS.matcher(map[pos[0]][pos[1] + 1]);
        if (pos[1] + 1 < map[0].length && m.find()) result.add(right);

        int[] down = new int[] {pos[0] + 1, pos[1], 2};
        m = ALLOWED_CHARS.matcher(map[pos[0] + 1][pos[1]]);
        if (pos[0] + 1 < map.length && m.find()) result.add(down);

        int[] left = new int[] {pos[0], pos[1] - 1, 3};
        m = ALLOWED_CHARS.matcher(map[pos[0]][pos[1] - 1]);
        if (pos[1] - 1 >= 0 && m.find()) result.add(left);

        return result;
    }

    public void DFS(int[] start, int[] end, HashSet<String> visited, ArrayList<String> result, ArrayList<ArrayList<String>> sequences) {
        if (start[0] == end[0] && start[1] == end[1]) {
            result.add(map[end[0]][end[1]]);
            sequences.add(result);

            return;
        }

        Matcher m = SEQUENTIABLE_CHARS.matcher(map[start[0]][start[1]]);
        if (m.find()) result.add(map[start[0]][start[1]]);
        for (int[] child: getSuccessors(start)) {
            String currentChild = Arrays.toString(new int[] { child[0], child[1] });
            if(!visited.contains(currentChild)) {

                visited.add(currentChild);
                DFS(child, end, (HashSet<String>) visited.clone(), (ArrayList<String>) result.clone(), sequences);
            }
        }
    }

    public ArrayList<String> generateBinaryCombinations(int n) {
        double span = Math.pow(2, n);
        String pattern = "[0]*" + String.join("", Collections.nCopies(nBandits, "[1][0]*"));
        Pattern occupiedLocations = Pattern.compile(pattern);
        ArrayList<String> result = new ArrayList<>();
        for (int i = 0; i < span; i++) {
            String binaryLocations = Integer.toBinaryString(i);
            Matcher m = occupiedLocations.matcher(binaryLocations);
            String prefix = String.join("", Collections.nCopies(n - binaryLocations.length(), "0"));
            if (m.matches()) result.add(prefix + binaryLocations);
        }
        Collections.reverse(result);

        return result;
    }

    public ArrayList<String> generateBinaryCombinations(int n, int filterOutBit) {
        ArrayList<String> combinations = generateBinaryCombinations(n);
        int i = 0;
        //Commented in order to generate all combinations. Filtering in NFCs and Utilities.
//        while (i < combinations.size()) {
//            if (combinations.get(i).charAt(filterOutBit) == '1') combinations.remove(i);
//            else i++;
//        }

        return combinations;
    }

    public String constructPairFromCombination(String combination) {
        int index = combination.indexOf("1");
        String allocation = "(";
        while (index >= 0) {
            allocation += "E" + (index + 1) + " : ";
            index = combination.indexOf("1", index + 1);
        }
        allocation = allocation.substring(0, allocation.length() - 3);
        allocation += ")";

        return allocation;
    }

    public void generateBanditsAllocations() {
        ArrayList<String> initialCombinations = generateBinaryCombinations(dangerousLocations);

        for (String combination: initialCombinations) {
            ArrayList<String> sequence = new ArrayList<>();
            sequence.add(constructPairFromCombination(combination));
            for (int i = 0; i < agentActions.size(); i++) banditsActions.add((ArrayList<String>) sequence.clone());
        }
    }

    public void generateBanditsFirstReactions() {
        int i = 0;
        for (ArrayList<String> banditsSequence: banditsActions) {
            ArrayList<String> initialLocations = new ArrayList<>(Arrays.asList(banditsSequence.get(0).replaceAll("[()]", "").split(" : ")));
            for (String agentSequence: agentActions.get(i % agentActions.size())) {
                if (agentSequence.contains("E")) {
                    if (initialLocations.contains(agentSequence)) {
                        banditsSequence.add("⚔ Attack ⚔");
                    } else {
                        banditsSequence.add("⚠ Alarm ⚠");
                    }
                    i++;
                    break;
                }
            }
        }
    }

    public void completeAttackOnlySequences() {
        int i = 0;
        for (ArrayList<String> banditsSequence: banditsActions) {
            if (banditsSequence.get(1).contains("Attack")) {
                ArrayList<String> initialLocations = new ArrayList<>(Arrays.asList(banditsSequence.get(0).replaceAll("[()]", "").split(" : ")));
                int skip = -1;
                for (String action: agentActions.get(i % agentActions.size())) {
                    if (action.contains("E") && skip++ >= 0) {
                        if (initialLocations.contains(action)) {
                            banditsSequence.add("⚔ Attack ⚔");
                        }
                    }
                }
            }
            i++;
        }
    }

    public void completeAlarmSequences() {
        Pattern dangerousLocationRegExp = Pattern.compile("E([0-9])");
        int i = 0;
        for (ArrayList<String> agentSequence: agentActions) {
            HashMap<String, ArrayList<ArrayList<String>>> mapping = new HashMap<>();

            int j = 0;
            for (ArrayList<String> banditsSequence: banditsActions) {
                if (j % agentActions.size() == i && banditsSequence.get(1).contains("Alarm")) {
                    ArrayList<ArrayList<String>> banditsSequencesForReallocation = new ArrayList<>();
                    ArrayList<String> combinations = new ArrayList<>();
                    for (String action: agentSequence) {
                        if (action.contains("E")) {
                            Matcher m = dangerousLocationRegExp.matcher(action);

                            if (m.matches()) {
                                combinations = generateBinaryCombinations(dangerousLocations, Integer.parseInt(m.group(1)) - 1);
                                break;
                            }
                        }
                    }
                    for (String combination: combinations) {
//                        System.out.println(combination);
                        ArrayList<String> banditSequenceBanditReallocationForAgentSequence = new ArrayList<>();
                        banditSequenceBanditReallocationForAgentSequence.add(constructPairFromCombination(combination));
                        for (String action: agentSequence) {
                            if (action.contains("E")) {
                                Matcher m = dangerousLocationRegExp.matcher(action);

                                if (m.matches()) {
                                    int dangerousLocationIndex = Integer.parseInt(m.group(1)) - 1;
                                    if (combination.charAt(dangerousLocationIndex) == '1') banditSequenceBanditReallocationForAgentSequence.add("⚔ Attack ⚔");
                                }
                            }
                        }
//                        System.out.println(Arrays.toString(banditSequenceBanditReallocationForAgentSequence.toArray()));
                        banditsSequencesForReallocation.add(banditSequenceBanditReallocationForAgentSequence);
                    }
                    mapping.put(banditsSequence.get(0), banditsSequencesForReallocation);
                }
                j++;
            }
            alarmMappings.add(mapping);
            i++;
        }
    }

    public void branchSequence(ArrayList<String> banditsSequence, int originalIndex) {
        ArrayList<ArrayList<String>> branches = new ArrayList<>();

        int attackIndex = banditsSequence.indexOf(BANDIT_ACTIONS.get("attack"));
        if (attackIndex >= 0) {
            // To show attacks, attackIndex + 1
            while (attackIndex != -1) {
                branches.add(new ArrayList<>(banditsSequence.subList(0, attackIndex)));
                branches.get(branches.size() - 1).add(BANDIT_ACTIONS.get("succeed"));


                banditsSequence.set(attackIndex, BANDIT_ACTIONS.get("fail"));
                attackIndex = banditsSequence.indexOf(BANDIT_ACTIONS.get("attack"));
            }
            branches.add(new ArrayList<>(banditsSequence));
        } else branches.add(banditsSequence);

        for (ArrayList<String> branch: branches) {
            banditsCompleteActions.add(branch);
            banditToAgentActionList.add(new HashSet<>());
            banditToAgentActionList.get(banditToAgentActionList.size() - 1).add(originalIndex % agentActions.size());
        }
    }

    public void mergeAllBanditsSequences() {
        for (int i = 0; i < agentActions.size(); i++) {
            for (int j = 0; j < banditsActions.size(); j++) {
                ArrayList<String> banditsSequence = banditsActions.get(j);
                String initialLocations = banditsSequence.get(0);
                String alarmAttack = banditsSequence.get(1);

                if (alarmAttack.contains(BANDIT_ACTIONS.get("alarm"))) {
                    if (alarmMappings.get(i).containsKey(initialLocations)) {
                        for (ArrayList<String> alarmSequence: alarmMappings.get(i).get(initialLocations)) {
                            ArrayList<String> completeBanditSequence = (ArrayList<String>) banditsSequence.clone();
                            completeBanditSequence.addAll(alarmSequence);
//                            System.out.println(Arrays.toString(completeBanditSequence.toArray()));

                            branchSequence(completeBanditSequence, i);
                        }
                    }
                } else branchSequence(banditsSequence, j);
            }
        }
    }

    public boolean hasValidReallocation(String comb1, String comb2) {
        String[] initialAllocation = comb1.replaceAll("[()]", "").split(" : ");
        String[] reAllocation = comb2.replaceAll("[()]", "").split(" : ");

        for (String Ei: initialAllocation) {
            for (String Ej: reAllocation) {
                if (Ei.equals(Ej)) return false;
            }
        }

        return true;
    }

    public void cleanBanditsSequences() {
        HashMap<String, Integer> uniqueBanditsSequences = new HashMap<>();
        int i = 0;
        while (i < banditsCompleteActions.size()) {
            ArrayList<String> sequence = banditsCompleteActions.get(i);
//            System.out.println(Arrays.toString(sequence.toArray()));
            int alarmIndex = sequence.indexOf(BANDIT_ACTIONS.get("alarm"));
            if (alarmIndex >= 0) {
                if (hasValidReallocation(sequence.get(0), sequence.get(alarmIndex + 1))) {
                    banditsCompleteActions.remove(i);
                    banditToAgentActionList.remove(i);
                    i--;
                }
                // do not remove if want to show alarm string
                sequence.remove(1);
                // do not remove if want to show alarm string
            }
            i++;
        }

        int j = 0;
        while (j < banditsCompleteActions.size()) {
            ArrayList<String> banditSequence = banditsCompleteActions.get(j);
            String sequenceCode = Arrays.toString(banditSequence.toArray()).replaceAll("[\\[\\]]", "");
            if (!uniqueBanditsSequences.containsKey(sequenceCode)) {
                uniqueBanditsSequences.put(sequenceCode, j);
            } else {
                for (Integer si: banditToAgentActionList.get(j)) {
                    banditToAgentActionList.get(uniqueBanditsSequences.get(sequenceCode)).add(si);
                }
                banditsCompleteActions.remove(j);
                banditToAgentActionList.remove(j);
                j--;
            }
            j++;
        }

        Integer[] proxyLookup = new Integer[banditsCompleteActions.size()];
        for (int k = 0; k < banditsCompleteActions.size(); k++) proxyLookup[k] = k;

        Arrays.sort(proxyLookup, new Comparator<Integer>() {
            public int compare(Integer a, Integer b) {
                ArrayList<String> sequenceA = banditsCompleteActions.get(a), sequenceB = banditsCompleteActions.get(b);
                int diff = sequenceA.size() - sequenceB.size();
                String sA = Arrays.toString(sequenceA.toArray());
                String sB = Arrays.toString(sequenceB.toArray());
                return diff < 0 ? -1 : (diff > 0 ? 1 : sA.compareTo(sB));
            }
        });

        ArrayList<ArrayList<String>> bc = new ArrayList<>();
        ArrayList<HashSet<Integer>> bta = new ArrayList<>();
        for(int k = 0; k < banditsCompleteActions.size(); k++) {
            bc.add(banditsCompleteActions.get(proxyLookup[k]));
            bta.add(banditToAgentActionList.get(proxyLookup[k]));
        }
        banditsCompleteActions = bc;
        banditToAgentActionList = bta;
    }

    private void setBanditsNoNature() {
        HashSet<String> unique = new HashSet<>();
        for (ArrayList<String> qi: banditsCompleteActions) {
            String cleanSequence = "";
            for (String ai: qi) {
                if (ai.contains(BANDIT_ACTIONS.get("fail")) || ai.contains(BANDIT_ACTIONS.get("succeed"))) continue;
                cleanSequence += ai + ", ";
            }
            cleanSequence = cleanSequence.substring(0, cleanSequence.length() - 2);
            completeToNoNature.put(Arrays.toString(qi.toArray()), "[" + cleanSequence + "]");
            unique.add(cleanSequence);
        }

        int i = 0;
        for (String sequence: unique) {
            banditsNoNature.add(new ArrayList<>());
            for (String ai: sequence.split(", ")) {
                banditsNoNature.get(i).add(ai);
            }
            i++;
        }
    }

    public void generateBanditsStrategies() {
        generateBanditsFirstReactions();
        completeAttackOnlySequences();
        completeAlarmSequences();
        mergeAllBanditsSequences();
        cleanBanditsSequences();
        setBanditsNoNature();
    }

    public void generateBanditInformationSets() {
        String attack = BANDIT_ACTIONS.get("attack");
        String alarm = BANDIT_ACTIONS.get("alarm");
        String succeed = BANDIT_ACTIONS.get("succeed");
        String fail = BANDIT_ACTIONS.get("fail");
        int i = 0;
        for (ArrayList<String> qi: banditsCompleteActions) {
            int j = 0;
            for (String action: qi) {
                if (!(attack.equals(action) || alarm.equals(action))) {
                    if (banditInformationSets.size() <= j) banditInformationSets.add(new HashMap<>());
                    if (action.equals(fail)|| action.equals(succeed)) {
                        if (!banditInformationSets.get(j).containsKey(attack)) banditInformationSets.get(j).put(attack, new HashSet<>());
                        banditInformationSets.get(j).get(attack).add(action);
                    } else {
                        if (!banditInformationSets.get(j).containsKey(alarm)) banditInformationSets.get(j).put(alarm, new HashSet<>());
                        banditInformationSets.get(j).get(alarm).add(action);
                    }
                    j++;
                }
            }
            i++;
        }
    }

    public void generateAgentInformationSets() {
        int i = 0;
        for (ArrayList<String> si: agentActions) {
            int j = 0;
            for (String action: si) {
                if (agentInformationSets.size() <= j) agentInformationSets.add(new HashSet<>());
                agentInformationSets.get(j).add(action);
                j++;
            }
            i++;
        }
    }

    public void generateInformationSets() {
        generateAgentInformationSets();
        generateBanditInformationSets();
    }

    public void setAgentNFCs() {
        int i = 0;
        ArrayList<HashSet<String>> anfc = new ArrayList<>();
        for (HashSet<String> ais: agentInformationSets) {
            anfc.add(new HashSet<>());
            for (String actionForSi: ais) {
                for (ArrayList<String> si: agentActions) {
                    if (i < si.size() && actionForSi.equals(si.get(i))) {
                        int k = 0;
                        String result = "r[";
                        for (String ai: si) {
                            if (k <= i) result += ai + ", ";
                            k++;
                        }
                        result = result.substring(0, result.length() - 2) + "]";
                        anfc.get(i).add(result);
                    }
                }
            }
//            System.out.println("Is" + (j + 1) + ": " + Arrays.toString(anfc.get(j).toArray()));
            i++;
        }
        agentNFCs.put("1", anfc.get(1));
        for (i = 2; i < anfc.size(); i++) {
            for (String ineq1: anfc.get(i)) {
                for (String ineq2: anfc.get(ineq1.split(", ").length - 2)) {
                    String match = ineq2.substring(0, ineq2.length() - 1);
                    match = match.replaceAll("\\[", "\\\\\\[");
                    match = match.replaceAll("\\(", "\\\\\\(");
                    match = match.replaceAll("\\)", "\\\\\\)");

                    Pattern rGrab = Pattern.compile("^" + match);

                    Matcher m = rGrab.matcher(ineq1);
                    if (m.find()) {
                        if (!agentNFCs.containsKey(ineq2)) agentNFCs.put(ineq2, new HashSet<>());
                        agentNFCs.get(ineq2).add(ineq1);
                    }
                }
            }
        }
    }

    private void setBanditNFCs() {
        int i = 0;
        int j = 0;
        ArrayList<HashSet<String>> bnfc = new ArrayList<>();
        for (HashMap<String, HashSet<String>> bis: banditInformationSets) {
            for (String key: bis.keySet()) {
                bnfc.add(new HashSet<>());
                for (String actionForQi: bis.get(key)) {
                    for (ArrayList<String> qi: banditsCompleteActions) {
                        if (i < qi.size() && actionForQi.equals(qi.get(i))) {
                            int k = 0;
                            String result = "r[";
                            for (String ai: qi) {
                                if (k <= i) result += ai + ", ";
                                k++;
                            }
                            result = result.substring(0, result.length() - 2) + "]";
                            bnfc.get(j).add(result);
                        }
                    }
                }
//                System.out.println("Iq" + (j + 1) + ": " + Arrays.toString(bnfc.get(j).toArray()));
                j++;
            }
            i++;
        }
        banditsNFCs.put("1", new HashMap<>());
        banditsNFCs.get("1").put("1", bnfc.get(0));
        for (String alarmPosition: alarmTriggers) {
            for (i = 1; i < bnfc.size(); i++) {
                for (String ineq1: bnfc.get(i)) {
                    for (String ineq2: bnfc.get(ineq1.split(", ").length - 2)) {
                        String[] moves = ineq1.split(", ");
                        if (hasAlarm(ineq1) && (moves[1].contains(alarmPosition) || moves[0].contains(alarmPosition))) break;

                        String match = ineq2.substring(0, ineq2.length() - 1);
                        match = match.replaceAll("\\[", "\\\\\\[");
                        match = match.replaceAll("\\(", "\\\\\\(");
                        match = match.replaceAll("\\)", "\\\\\\)");

                        Pattern rGrab = Pattern.compile("^" + match);

                        Matcher m = rGrab.matcher(ineq1);
                        if (m.find() && !(ineq1.contains(BANDIT_ACTIONS.get("fail")) || ineq1.contains(BANDIT_ACTIONS.get("succeed"))) &&
                                !isIllegalQ(new ArrayList<>(Arrays.asList(ineq1.split(", "))))) {
                            if (!banditsNFCs.containsKey(ineq2)) banditsNFCs.put(ineq2, new HashMap<>());
                            if (!banditsNFCs.get(ineq2).containsKey(alarmPosition)) banditsNFCs.get(ineq2).put(alarmPosition, new HashSet<>());
                            banditsNFCs.get(ineq2).get(alarmPosition).add(ineq1);
                        }
                    }
                }
            }
        }
    }

    private void setDifferentialNFCs() {
        for (String equals: banditsNFCs.keySet()) {

            for (String alarmPosition1: banditsNFCs.get(equals).keySet()) {
                if (alarmPosition1.equals("1")) continue;
                for (String r: banditsNFCs.get(equals).get(alarmPosition1)) {
                    boolean isTheSameInAll = true;
                    for (String alarmPosition2 : banditsNFCs.get(equals).keySet()) {
                        isTheSameInAll &= banditsNFCs.get(equals).get(alarmPosition2).contains(r);
                        if (!isTheSameInAll) break;
                    }
                    if (!differentialNFCs.containsKey(alarmPosition1)) differentialNFCs.put(alarmPosition1, new HashSet<>());
                    differentialNFCs.get(alarmPosition1).add(r);
                }
            }
        }

//        for (String alarmPosition1: differentialNFCs.keySet()) {
//            for (String alarmPosition2: differentialNFCs.keySet()) {
//                if (alarmPosition1.equals(alarmPosition2)) continue;
//                differentialNFCs.get(alarmPosition1).retainAll(differentialNFCs.get(alarmPosition2));
//            }
//            System.out.println(alarmPosition1 + ": " + differentialNFCs.get(alarmPosition1));
//        }

//        for (String alarmPosition: differentialNFCs.keySet()) {
//            for (String r: differentialNFCs.get(alarmPosition)) {
//                String alarmedId = "[" + alarmPosition + "]:" + r;
//                if (!banditsNFCs.containsKey(r)) banditsNFCs.put(r, new HashMap<>());
//                if (!banditsNFCs.get(r).containsKey(alarmedId)) banditsNFCs.get(r).put(alarmedId, new HashSet<>());
//                banditsNFCs.get(r).get(alarmedId).add(alarmedId);
//            }
//        }
    }

    private void setNetworkFlowConstraints() {
        setAgentNFCs();
        setBanditNFCs();
        setDifferentialNFCs();
    }

    private int nActionsForSequence(ArrayList<String> si) {
        int max = 0;
        for (ArrayList<String> sj: agentActions) {
            boolean sameSequence = true;
            int j = 0;
            int count = 0;
            for (int i = 0; i < si.size(); i++) {
                String ai = si.get(i);
                String aj = sj.get(i + j);
                sameSequence &= aj.equals(ai);
                if (ai.contains("G")) {
                    j--;
                    continue;
                } else if (aj.contains("G")) {
                    i--;
                    j++;
                    continue;
                } else if (aj.equals(ai)) count++;
                else {
                    max = max < count && !sameSequence ? count : max;
                    break;
                }
            }
        }

        return max;
    }

    private boolean hasAlarm(ArrayList<String> q) {
        return Arrays.toString(q.toArray()).split("\\(.*?\\)").length >= 3;
    }

    private boolean hasAlarm(String q) {
        return q.split("\\(.*?\\)").length >= 3;
    }

    private String getFirstDangerous(ArrayList<String> s, ArrayList<String> q) {
        for (String action: s) {
            if (action.contains("E")) return action;
        }
        return "";
    }

    private void setExpectedUtilityIdentifiersForAttacker(ArrayList<String> s, ArrayList<String> q) throws IloException {
        String sKey = Arrays.toString(s.toArray());
        String qKey = Arrays.toString(q.toArray());
        String alarmPosition = "";

        boolean hasAlarm = hasAlarm(q);
        if (hasAlarm) alarmPosition = getFirstDangerous(s, q);

        int succeedAttack = q.indexOf(BANDIT_ACTIONS.get("succeed"));
        int encounters = 0;
        int dangerousVisited = 0;
        int nActions = nActionsForSequence(s);

        // Return immediately if the utility is zero, or if the sequences are illegal
        if (succeedAttack >= 0 || !utilities.containsKey(sKey) || !utilities.get(sKey).containsKey(completeToNoNature.get(qKey))) return;

        String key = "X";
        if (!agentUtilities.containsKey(key)) agentUtilities.put(key, new HashMap<>());
        if (!agentUtilities.get(key).containsKey(key)) agentUtilities.get(key).put(key, new ArrayList<>());
        if (agentUtilities.get(key).get(key).size() == 0) agentUtilities.get(key).get(key).add("X");
        for (int i = 1; i < s.size(); i++) {
            boolean dangerous = s.get(i).contains("E");
            boolean empty = !q.get(!hasAlarm ? 0 : 1).contains(s.get(i));
            boolean dangerousAndEmpty = dangerous && empty;
            if (dangerous) dangerousVisited++;
            if (dangerous && !empty) encounters++;
            if (!agentUtilities.containsKey(key)) agentUtilities.put(key, new HashMap<>());
            String sAction = s.get(i);
            String utility = "";
            String r = "r[";
            for (String qAction: q) {
                if (!(qAction.equals(BANDIT_ACTIONS.get("fail")))) r += qAction + ", ";
            }
            r = r.substring(0, r.length() - 2) + "]";

            // This is for identifying pairs of [(initial_allocations), (reallocations)] that belong to different information sets of the bandit
            if (hasAlarm && differentialNFCs.containsKey(alarmPosition) && differentialNFCs.get(alarmPosition).contains(r)) r = "[" + alarmPosition + "]:" + r;
            r = "*" + r;

            utility += utilities.get(sKey).get(completeToNoNature.get(qKey))  + r;
            if (!agentUtilities.get(key).containsKey(sAction)) agentUtilities.get(key).put(sAction, new ArrayList<>());
            agentUtilities.get(key).get(sAction).add(utility);

            key = key + "_Iq" + sAction + (dangerousAndEmpty ? "!" : "");

//            if ((encounters == nBandits || dangerousVisited >= nActions) && dangerousVisited > 1) break;
        }
    }

    private void setExpectedUtilityIdentifiersForAgent(ArrayList<String> s, ArrayList<String> q) throws IloException {
        String sKey = Arrays.toString(s.toArray());
        String qKey = Arrays.toString(q.toArray());
        String alarmPosition = "";

        boolean hasAlarm = hasAlarm(q);
        String firstE = getFirstDangerous(s, q);

        int succeedAttack = q.indexOf(BANDIT_ACTIONS.get("succeed"));
        int encounters = 0;
        int dangerousVisited = 0;
        int nActions = nActionsForSequence(s);

        // Return immediately if the utility is zero, or if the sequences are illegal
        if (succeedAttack >= 0 || !utilities.containsKey(sKey) || !utilities.get(sKey).containsKey(completeToNoNature.get(qKey))) return;
        String key = "Y";
        if (!banditUtilities.containsKey(key)) banditUtilities.put(key, new HashMap<>());
        if (!banditUtilities.get(key).containsKey(key)) banditUtilities.get(key).put(key, new ArrayList<>());
        if (banditUtilities.get(key).get(key).size() == 0) banditUtilities.get(key).get(key).add("Y");
        for (int i = 0; i < q.size(); i++) {
            String qAction = q.get(i);
            if(qAction.equals(BANDIT_ACTIONS.get("fail"))) continue;

            if (!banditUtilities.containsKey(key)) banditUtilities.put(key, new HashMap<>());
            String utility = "";
            String r = "r[";
            for (String sAction: s) {
                if (sAction.contains("E") || sAction.contains("G")) r += sAction + ", ";
            }
            r = r.substring(0, r.length() - 2) + "]";

            // This is for identifying pairs of [(initial_allocations), (reallocations)] that belong to different information sets of the bandit
            if (hasAlarm && differentialNFCs.containsKey(alarmPosition) && differentialNFCs.get(alarmPosition).contains(r)) r = "[" + alarmPosition + "]:" + r;
            r = "*" + r;

            utility += utilities.get(sKey).get(completeToNoNature.get(qKey))  + r;
            if (!banditUtilities.get(key).containsKey(qAction)) banditUtilities.get(key).put(qAction, new ArrayList<>());
            banditUtilities.get(key).get(qAction).add(utility);

            key = key + "_Iq" + qAction + "_" + firstE + (hasAlarm ? "!" : "");

//            if ((encounters == nBandits || dangerousVisited >= nActions) && dangerousVisited > 1) break;
        }
    }

    private void replaceSubExpectedUtilities() {
        Comparator<String> cmp = new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                int diff = a.length() - b.length();
                return diff == 0 ? a.compareTo(b) : diff;
            }
        };

        List<String> sortedKeys = Arrays.asList(agentUtilities.keySet().toArray(new String[agentUtilities.size()]));
        Collections.sort(sortedKeys, cmp);
        int i = 0;
        for (String v : sortedKeys) {
            for (String actionKeyV : agentUtilities.get(v).keySet()) {
                ArrayList<String> s1 = agentUtilities.get(v).get(actionKeyV);
                for (String u : agentUtilities.keySet()) {
                    for (String actionKeyU : agentUtilities.get(u).keySet()) {
                        if (u.equals(v) && actionKeyV.equals(actionKeyU)) continue;

                        ArrayList<String> s2 = agentUtilities.get(u).get(actionKeyU);
                        int lastI = u.lastIndexOf("_") == - 1 ? u.length() : u.lastIndexOf("_");

                        if (v.equals(u.substring(0, lastI)) && s1.containsAll(s2)) {
                            for (String action : s2) agentUtilities.get(v).get(actionKeyV).remove(action);
                            if (!agentUtilities.get(v).get(actionKeyV).contains(u)) agentUtilities.get(v).get(actionKeyV).add(u);
                        }
                    }
                }
            }
            i++;
        }

        sortedKeys = Arrays.asList(banditUtilities.keySet().toArray(new String[banditUtilities.size()]));
        Collections.sort(sortedKeys, cmp);
        i = 0;
        for (String v : sortedKeys) {
            for (String actionKeyV : banditUtilities.get(v).keySet()) {
                ArrayList<String> s1 = banditUtilities.get(v).get(actionKeyV);
                for (String u : banditUtilities.keySet()) {
                    for (String actionKeyU : banditUtilities.get(u).keySet()) {
                        if (u.equals(v) && actionKeyV.equals(actionKeyU)) continue;

                        ArrayList<String> s2 = banditUtilities.get(u).get(actionKeyU);
                        int lastI = u.lastIndexOf("_") == - 1 ? u.length() : u.lastIndexOf("_");

                        if (v.equals(u.substring(0, lastI)) && s1.containsAll(s2)) {
                            for (String action : s2) banditUtilities.get(v).get(actionKeyV).remove(action);
                            if (!banditUtilities.get(v).get(actionKeyV).contains(u)) banditUtilities.get(v).get(actionKeyV).add(u);
                        }
                    }
                }
            }
            i++;
        }
    }

    private boolean isIllegal(ArrayList<String> s, ArrayList<String> q) {
        boolean hasAlarm = Arrays.toString(q.toArray()).split("\\(.*?\\)").length >= 3;
        for (String ai: s) {
            if (ai.contains("E")) {
                if (hasAlarm && (q.get(0).contains(ai) || q.get(1).contains(ai))) return true;
                else if (!hasAlarm && !q.get(0).contains(ai)) return true;
                return false;
            }
        }

        return false;
    }

    private boolean isIllegalQ(ArrayList<String> q) {
        boolean hasAlarm = Arrays.toString(q.toArray()).split("\\(.*?\\)").length >= 3;
        boolean result = true;
        for (ArrayList<String> s : agentActions) {
            result &= isIllegal(s, q);
        }

        return hasAlarm && result;
    }

    private void setExpectedUtilityFunctions(ArrayList<String> s, ArrayList<String> q) {
        if (isIllegal(s, q)) return;

        String sKey = Arrays.toString(s.toArray());
        String qKey = Arrays.toString(q.toArray());

        int goldCount = (int) Arrays.toString(s.toArray()).chars().filter(ch -> ch == 'G').count();
        int goldCollected = goldCount * SCORES.get(AGENT_ACTIONS.get("gold"));
        int failAttacks = Arrays.toString(q.toArray()).split(BANDIT_ACTIONS.get("fail")).length - 1;
        double total = 0;
        total = (Math.pow(1 - pSuccessfulAttack, failAttacks)*(10 + goldCollected));

        if (!utilities.containsKey(sKey)) utilities.put(sKey, new HashMap<>());
        if (!utilities.get(sKey).containsKey(completeToNoNature.get(qKey)) || utilities.get(sKey).get(completeToNoNature.get(qKey)) > total) utilities.get(sKey).put(completeToNoNature.get(qKey), total);
    }

    private void constructExpectedUtilityFunctions() throws IloException {
        int j = 0;
        for (ArrayList<String> qj: banditsCompleteActions) {
            int i = 0;
            for (ArrayList<String> si : agentActions) {
                if (banditToAgentActionList.get(j).contains(i)) {
                    setExpectedUtilityFunctions(si, qj);
                    setExpectedUtilityIdentifiersForAgent(si, qj);
                    setExpectedUtilityIdentifiersForAttacker(si, qj);
                }
                i++;
            }
            j++;
        }

        replaceSubExpectedUtilities();
    }

    private void setCPLEXLinearEq() throws IloException {
        ArrayList<IloNumVar> eq = new ArrayList<>();
        for (String keyInformationSet: agentUtilities.keySet()) {
            for (String keyActionTaken: agentUtilities.get(keyInformationSet).keySet()) {
                if (!cplexVars.containsKey(keyInformationSet)) cplexVars.put(keyInformationSet, cplex.numVar(0, Float.MAX_VALUE, IloNumVarType.Float, keyInformationSet));
                IloNumVar V = cplexVars.get(keyInformationSet);

                IloNumExpr sum = cplex.numExpr();
                for (String utility: agentUtilities.get(keyInformationSet).get(keyActionTaken)) {
                    String[] u = utility.split("\\*");
                    IloNumExpr num = cplex.constant(u.length == 2 ? Float.parseFloat(u[0]) : 1);

                    if (!cplexVars.containsKey(u[u.length == 2 ? 1 : 0])) cplexVars.put(u[u.length == 2 ? 1 : 0], cplex.numVar(0, Float.MAX_VALUE, IloNumVarType.Float, u[u.length == 2 ? 1 : 0]));
                    IloNumExpr r = cplexVars.get(u[u.length == 2 ? 1 : 0]);
                    sum = cplex.sum(cplex.prod(num, r), sum);
                }
                cplex.addGe(V, sum);
                eq.add(V);
            }
        }
        IloNumExpr sum = cplex.numExpr();
        sum = cplex.sum(cplexVars.get("X"), sum);

        cplex.addMinimize(sum);


        eq = new ArrayList<>();
        for (String keyInformationSet: banditUtilities.keySet()) {
            for (String keyActionTaken: banditUtilities.get(keyInformationSet).keySet()) {
                if (!cplexVars.containsKey(keyInformationSet)) cplexVars.put(keyInformationSet, cplex.numVar(0, Float.MAX_VALUE, IloNumVarType.Float, keyInformationSet));
                IloNumVar V = cplexVars.get(keyInformationSet);

                sum = cplex.numExpr();
                for (String utility: banditUtilities.get(keyInformationSet).get(keyActionTaken)) {
                    String[] u = utility.split("\\*");
                    IloNumExpr num = cplex.constant(u.length == 2 ? Float.parseFloat(u[0]) : 1);

                    if (!cplexVars.containsKey(u[u.length == 2 ? 1 : 0])) cplexVars.put(u[u.length == 2 ? 1 : 0], cplex.numVar(0, Float.MAX_VALUE, IloNumVarType.Float, u[u.length == 2 ? 1 : 0]));
                    IloNumExpr r = cplexVars.get(u[u.length == 2 ? 1 : 0]);
                    sum = cplex.sum(cplex.prod(num, r), sum);
                }
                cplex.addLe(V, sum);
                eq.add(V);
            }
        }
        sum = cplex.numExpr();
        sum = cplex.sum(cplexVars.get("Y"), sum);

//        cplex.addMaximize(sum);
    }

    private double getValueOfTheGame() throws IloException {
        Comparator<String> cmp = new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                int diff = a.length() - b.length();
                return diff == 0 ? a.compareTo(b) : diff;
            }
        };

        List<String> sortedKeys = Arrays.asList(agentNFCs.keySet().toArray(new String[agentNFCs.size()]));
        Collections.sort(sortedKeys, cmp);
        for (String key: sortedKeys) {
            if (!cplexVars.containsKey(key)) cplexVars.put(key, cplex.numVar(0, 1, IloNumVarType.Float, key));
            IloNumVar V = cplexVars.get(key);
            IloNumExpr sum = cplex.numExpr();
            for (String nfc: agentNFCs.get(key)) {
                if (!cplexVars.containsKey(nfc)) cplexVars.put(nfc, cplex.numVar(0, 1, IloNumVarType.Float, nfc));
                IloNumVar r = cplexVars.get(nfc);
                sum = cplex.sum(sum, r);
            }
            cplex.addEq(key.matches("\\d") ? cplex.constant(Float.parseFloat(key)) : V, sum);
        }

        sortedKeys = Arrays.asList(banditsNFCs.keySet().toArray(new String[banditsNFCs.size()]));
        Collections.sort(sortedKeys, cmp);
        for (String key: sortedKeys) {
            if (!cplexVars.containsKey(key)) cplexVars.put(key, cplex.numVar(0, 1, IloNumVarType.Float, key));
            IloNumVar V = cplexVars.get(key);

            for (String alarmPosition: banditsNFCs.get(key).keySet()) {
                IloNumExpr sum = cplex.numExpr();
                for (String nfc: banditsNFCs.get(key).get(alarmPosition)) {
                    String apNFC = nfc;
                    if (differentialNFCs.containsKey(alarmPosition) &&
                            differentialNFCs.get(alarmPosition).contains(nfc)) apNFC = "[" + alarmPosition + "]:" + nfc;
                    if (!cplexVars.containsKey(apNFC)) cplexVars.put(apNFC, cplex.numVar(0, 1, IloNumVarType.Float, apNFC));
                    IloNumVar r = cplexVars.get(apNFC);
                    sum = cplex.sum(sum, r);
                }
                cplex.addEq(key.matches("\\d") ? cplex.constant(Float.parseFloat(key)) : V, sum);
            }
        }
        setCPLEXLinearEq();
        cplex.solve();
        cplex.exportModel("hilarjos.lp");

        return cplex.getValue(cplexVars.get("X"));
//        return cplex.getValue(cplexVars.get("Y"));
    }

    public String[][] getMap() { return map; }
    public ArrayList<ArrayList<String>> getAgentActions() { return agentActions; }
    public ArrayList<HashSet<String>> getAgentInformationSets() { return agentInformationSets; }
    public HashMap<String, HashSet<String>> getAgentNFCs() { return agentNFCs; }
    public HashMap<String, HashMap<String, ArrayList<String>>> getAgentUtilities() { return agentUtilities; }
    public HashMap<String, HashMap<String, ArrayList<String>>> getBanditUtilities() { return banditUtilities; }
    public ArrayList<ArrayList<String>> getBanditsActions() { return banditsCompleteActions; }
    public ArrayList<HashMap<String, HashSet<String>>> getBanditsInformationSets() { return banditInformationSets; }
    public HashMap<String, HashMap<String, HashSet<String>>> getBanditsNFCs() { return banditsNFCs; }
    public ArrayList<HashSet<Integer>> getBanditToAgentActionList() { return banditToAgentActionList; }
    public HashMap<String, HashMap<String, Double>> getUtilities() { return utilities; }
    public ArrayList<ArrayList<String>> getBanditsNoNature() { return banditsNoNature; }

    private void removeDuplicates() {
        HashSet<String> unique = new HashSet<>();
        for (int i = 0; i < agentActions.size(); i++) {
            ArrayList<String> sequence = agentActions.get(i);
            String id = Arrays.toString(sequence.toArray());
            if (unique.contains(id)) {
                agentActions.remove(i);
                i--;
            } else unique.add(id);
        }
    }

    private void setAlarmTriggers() {
        for (ArrayList<String> si: agentActions) {
            for (String ai: si) {
                if (ai.contains("E")) {
                    alarmTriggers.add(ai);
                    break;
                }
            }
        }
    }

    public double runFor(String env, String path) throws IloException {
        if (!env.equals("dataset")) return 0;

        cplex = new IloCplex();
        readInputFromDataset(path);
        DFS(getStart(), getEnd(), new HashSet<>(), new ArrayList<>(), agentActions);
        setAlarmTriggers();
        removeDuplicates();
        generateBanditsAllocations();
        generateBanditsStrategies();
        generateInformationSets();
        setNetworkFlowConstraints();
        constructExpectedUtilityFunctions();

        return getValueOfTheGame();
    }

    public double runFor(String env) throws IloException {
        if (!env.equals("cli")) return 0;

        cplex = new IloCplex();
        readInputFromCLI();
        DFS(getStart(), getEnd(), new HashSet<>(), new ArrayList<>(), agentActions);
        setAlarmTriggers();
        removeDuplicates();
        generateBanditsAllocations();
        generateBanditsStrategies();
        generateInformationSets();
        setNetworkFlowConstraints();
        constructExpectedUtilityFunctions();

        return getValueOfTheGame();
    }

    public static void test() throws IloException {
        for (int i = 1; i <= 5; i++) {
            long startTime = System.nanoTime();

            Main s = new Main();
            String filename = BASE_PATH + "pub" + (i >= 10 ? i : "0" + i);
            double result = s.runFor("dataset", filename + ".in");
            double expected = s.readOutputFromDataset(filename + ".out");

//            Print.Game(s);

            long stopTime = System.nanoTime();
            System.out.println("total time: " + (stopTime - startTime)/ 1000000000.0);
            String checkMark = Math.abs(result - expected) < .0001  ? "✔" : "✘";
            System.out.println(checkMark + " - { expected: " + Double.toString(expected) + ", result: " + Double.toString(result) + " };\r\n");
        }
    }

    public static void conquer() throws IloException {
        Main s = new Main();
        double result = s.runFor("cli");
        Print.Game(s);

        System.out.println("SOLUTION_VALUE: " + result);
    }

    public static void main(String[] args) throws IloException {
//        Main.test();
        Main.conquer();
    }
}
