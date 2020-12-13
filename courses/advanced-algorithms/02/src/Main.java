package pal;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Main {
    private ArrayList<ArrayList<Integer>> adjList = new ArrayList<>(1000000);
    private ArrayList<ArrayList<Integer>> adjListReversed = new ArrayList<>(1000000);
    private ArrayList<ArrayList<Integer>> condensedList = new ArrayList<>();
    private ArrayList<Integer> sccMemberships = new ArrayList<>(1000000); // {node => scc membership}
    private ArrayList<ArrayDeque<Integer>> sccList = new ArrayList<>();
    private boolean[] discovered = new boolean[1000000];
    private ArrayList<Integer> wayfarers = new ArrayList<>(1000);
    private int destination;
    private ArrayDeque<Integer> nodeClosedTimesStack = new ArrayDeque<>(1000000); // node
    private int sccCount = 0;
    private ArrayList<HashSet<Integer>> markSCCs = new ArrayList<>(1000000);
    private ArrayList<Integer> sccIncomingMax = new ArrayList<>(1000000);
    private HashSet<Integer> representatives = new HashSet<>();
    private static String BASE_PATH = "/Users/ananias/Documents/CTU/Advanced Algorithms/Labs/02/dataset/";

    private void resetDiscovered() {
        Arrays.fill(discovered, false);
    }

    private void resetTimes() {
        nodeClosedTimesStack = new ArrayDeque<>();
    }

    private void sccResetCount() {
        sccCount = 0;
    }

    public void printSCC() {
        String output = "";
        int i = 0;
        for (ArrayDeque<Integer> scc: sccList) {
            output += "[ SCC #" + ++i + " contains " + scc.size() + " member(s), which are the nodes { ";
            for (Integer sccMember: scc) {
                output += "[ v: " + (sccMember + 1) + " ], ";
            }
            output = output.replaceAll(",\\s+$", "");
            output += " } ]\r\n";
        }
        System.out.println(output);
    }

    public void printWayfarersAndDestinationLocations() {
        String output = "";
        for (Integer v: wayfarers) {
            output += "[ Wayfarer node #" + (v + 1) + ": is at SCC #" + (sccMemberships.get(v) + 1) + " vicinity ]\r\n";
        }
        output += "[ Destination node #" + (destination + 1) + ": is at SCC #" + (sccMemberships.get(destination) + 1) + " vicinity ]\r\n";
        System.out.println(output);
    }

    public void printCondensedList() {
        String output = "";
        int i = 0;
        for (ArrayList<Integer> group: condensedList) {
            for (Integer j: group) {
                output += "[ SCC #" + ++i + " points to SCC #" + (j + 1)  + " ]\r\n";
            }
        }
        System.out.println(output);
    }

    public void readInputFromDataset(String path) {
        Iterator<String> lines;
        try {
            lines = Files.lines(Paths.get(BASE_PATH + path)).iterator();
            String[] input = lines.next().split(" ");
            int n = Integer.parseInt(input[0]), m = Integer.parseInt(input[1]), w = Integer.parseInt(input[2]), d = Integer.parseInt(input[3]);
            System.out.println("\r\n{ N: " + n + ", M: " + m + ", W: " + w + ", D node is: " + d + " }");

            destination = d - 1;
            for (int i = 0; i < n; i++) {
                adjList.add(i, new ArrayList<>());
                adjListReversed.add(i, new ArrayList<>());
                sccMemberships.add(i, -1);
            }

            input = lines.next().split(" ");
            for (int i = 0; i < w; i++) {
                wayfarers.add(i, Integer.parseInt(input[i]) - 1);
            }

            while (lines.hasNext()) {
                input = lines.next().split(" ");
                int v1 = Integer.parseInt(input[0]) - 1, v2 = Integer.parseInt(input[1]) - 1;
                adjList.get(v1).add(v2);
                adjListReversed.get(v2).add(v1);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int readOutputFromDataset(String path) {
        Iterator<String> lines;
        try {
            lines = Files.lines(Paths.get(BASE_PATH + path)).iterator();
            String[] input = lines.next().split(" ");
            return Integer.parseInt(input[0]);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return -1;
    }

    public void readInputFromCLI() {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String line = reader.readLine();
            StringTokenizer st = new StringTokenizer(line, " ");
            int n = Integer.parseInt(st.nextToken()), m = Integer.parseInt(st.nextToken()), w = Integer.parseInt(st.nextToken()), d = Integer.parseInt(st.nextToken());

            destination = d - 1;
            for (int i = 0; i < n; i++) {
                adjList.add(i, new ArrayList<>());
                adjListReversed.add(i, new ArrayList<>());
                sccMemberships.add(i, -1);
            }

            line = reader.readLine();
            st = new StringTokenizer(line, " ");
            for (int i = 0; i < w; i++) {
                wayfarers.add(Integer.parseInt(st.nextToken()) - 1);
            }

            for (int i = 0; i < m; i++) {
                line = reader.readLine();
                st = new StringTokenizer(line, " ");
                int v1 = Integer.parseInt(st.nextToken()) - 1, v2 = Integer.parseInt(st.nextToken()) - 1;
                adjList.get(v1).add(v2);
                adjListReversed.get(v2).add(v1);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void dfsSetup() {
        resetDiscovered();
        resetTimes();
    }

    public void dfsFromRoot(int v1) {
        if (discovered[v1]) return;

        discovered[v1] = true;
        for (Integer v2: adjList.get(v1)) {
            if (!discovered[v2]) dfsFromRoot(v2);
        }
        nodeClosedTimesStack.push(v1);
    }

    public void dfs() {
//        dfsSetup();
        for (int i = 0; i < adjList.size(); i++) {
            dfsFromRoot(i);
        }
    }

    public void sccSetup() {
        resetDiscovered();
        sccResetCount();
    }

    public ArrayDeque<Integer> sccFromRoot(int v1, ArrayDeque<Integer> sccMembers) {
        if (discovered[v1]) return sccMembers;

        discovered[v1] = true;
        for (Integer v2: adjListReversed.get(v1)) {
            if (!discovered[v2]) sccFromRoot(v2, sccMembers);
        }
        sccMembers.push(v1);
        sccMemberships.set(v1, sccCount);

        return sccMembers;
    }

    public void scc() {
        sccSetup();
        for (Integer v: nodeClosedTimesStack) {
            ArrayDeque<Integer> sccCurrent = sccFromRoot(v, new ArrayDeque<>());
            if (!sccCurrent.isEmpty()) {
                sccList.add(sccCurrent);
                sccCount++;
            }
        }
    }

    public void condenseSetup() {
        resetDiscovered();
        for (int i = 0; i < sccList.size(); i++) {
            condensedList.add(i, new ArrayList<>());
        }
    }

    public void condenseFromRoot(int v1) {
        if (discovered[v1]) return;

        discovered[v1] = true;
        for (Integer v2: adjList.get(v1)) {
            if (!discovered[v2]) condenseFromRoot(v2);
            if (!sccMemberships.get(v1).equals(sccMemberships.get(v2))) condensedList.get(sccMemberships.get(v1)).add(sccMemberships.get(v2));
        }
    }

    // make condense list point to SCCs
    public void condense() {
        condenseSetup();
        for (ArrayDeque<Integer> scc: sccList) {
            for (Integer v: scc) {
                condenseFromRoot(v);
            }
        }
    }

    public void dfsCondensedFromRoot(int v1) {
        if (discovered[v1]) return;

        discovered[v1] = true;
        for (Integer v2: condensedList.get(v1)) {
            if (!discovered[v2]) dfsCondensedFromRoot(v2);
        }
        nodeClosedTimesStack.push(v1);
    }

    public void topologicalSort() {
        resetDiscovered();
        for (int i = 0; i < condensedList.size(); i++) {
            if (condensedList.get(i).size() > 0) dfsCondensedFromRoot(i);
        }
    }

    public void travelSetup() {
        for (int i = 0; i < sccList.size(); i++) {
            markSCCs.add(i, new HashSet<>());
            sccIncomingMax.add(i, Integer.MIN_VALUE);
        }
    }

    public void setRepresentativesOfSCCs() {
        HashSet<Integer> uniqueSCCs = new HashSet<>();
        for (Integer w : wayfarers) {
            Integer scc = sccMemberships.get(w);
            if (!uniqueSCCs.contains(scc)) {
                uniqueSCCs.add(scc);
                representatives.add(w);
            }
        }
    }

    public void markRoad(Integer scc1, Integer w) {
        markSCCs.get(scc1).add(w);
        for (Integer scc2: condensedList.get(scc1)) {
            if (!markSCCs.get(scc2).contains(w)) markRoad(scc2, w);
        }
    }

    // Find longest / most expensive path (based on members of scc)
    public void selectBest(Integer sccW, Integer w) {
        markSCCs.get(sccW).add(w);
        for (int scc1 = sccW; scc1 < condensedList.size(); scc1++) {
            int scc1Size = sccList.get(scc1).size();
            int oldMax1 = sccIncomingMax.get(scc1);
            int newMax1 = oldMax1 > 0 ? oldMax1 : scc1Size;
            sccIncomingMax.set(scc1, newMax1);

            if (markSCCs.get(scc1).contains(w) && markSCCs.get(scc1).size() >= representatives.size()) {
                for (Integer scc2 : condensedList.get(scc1)) {
                    markSCCs.get(scc2).add(w);
                    if (scc1 != scc2 && markSCCs.get(scc2).size() >= representatives.size()) {
                        int scc2Size = sccList.get(scc2).size();
                        int oldMax2 = sccIncomingMax.get(scc2);
                        int newMax2 = sccIncomingMax.get(scc1) + scc2Size;
                        if (newMax2 > oldMax2) sccIncomingMax.set(scc2, newMax2);
                    }
                }
            }
        }
    }

    public int travel() {
        travelSetup();
        setRepresentativesOfSCCs();

        int i = 0;
        for (Integer w : representatives) {
            Integer scc = sccMemberships.get(w);
            if (++i != representatives.size()) {
                markRoad(scc, w);
            } else {
                selectBest(scc, w);
            }
        }

        return sccIncomingMax.get(sccMemberships.get(destination));
    }

    public int runFor(String env, String path) {
        if (!env.equals("dataset")) return 0;

        readInputFromDataset(path);
        dfs();
        scc();
        condense();
        topologicalSort();

//      printWayfarersAndDestinationLocations();
//      printSCC();
//      printTimesTopologicalSort();
//      printCondensedList();
        return travel();
//        return 0;
    }

    public int runFor(String env) {
        if (!env.equals("cli")) return 0;
        readInputFromCLI();
        dfs();
        scc();
        condense();
        topologicalSort();

//      printWayfarersAndDestinationLocations();
//      printSCC();
//      printTimesTopologicalSort();
//      printCondensedList();
        return travel();
    }

    public static void test() {
        for (int i = 1; i <= 10; i++) {
            Main s = new Main();
            long startTime = System.nanoTime();

            String filename = "pub" + (i >= 10 ? i : "0" + i);
            int result = s.runFor("dataset", filename + ".in");
            int expected = s.readOutputFromDataset(filename + ".out");

            long stopTime = System.nanoTime();
            System.out.println("total time: " + (stopTime - startTime)/ 1000000000.0);
            String checkMark = result == expected  ? "✔" : "✘";
            System.out.println(checkMark + " - {expected: " + expected + ", result:" + result + " };\r\n");
        }
    }

    public static void conquer() {
        Main s = new Main();
        System.out.println(s.runFor("cli"));
    }

    public static void main(String[] args) {
        // Main.test();
       Main.conquer();
    }
}
