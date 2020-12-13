package pal;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Main {
    private ArrayList<ArrayList<HashSet<Integer>>> adjLists = new ArrayList<>(250);
    private ArrayList<ArrayList<HashSet<Integer>>> loopAdjList = new ArrayList<>(250);
    private ArrayList<ArrayList<Boolean>> loopParents = new ArrayList<>(250);
    private ArrayList<ArrayList<Integer>> heights = new ArrayList<>(250);
    private ArrayList<int[]> roots = new ArrayList<>(250);
    private HashMap<String, Integer> isomorphisms = new HashMap<>();
    private int N_nodes;
    private int M_edges;
    boolean[][] inFIFO;
    boolean[][] removed;
    boolean[][] waiting;
    int[][] removedSiblings;

    private static String BASE_PATH = "/Users/ananias/Documents/CTU/Advanced Algorithms/Labs/03/dataset/";

    public ArrayList<HashSet<Integer>> initializedIntegerAdjList(int size) {
        ArrayList<HashSet<Integer>> adjList = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            adjList.add(i, new HashSet<>());
        }

        return adjList;
    }

    public ArrayList<Boolean> initializedBooleanList(int size) {
        ArrayList<Boolean> adjList = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            adjList.add(i, false);
        }

        return adjList;
    }

    public ArrayList<Integer> initializedIntegerList(int size) {
        ArrayList<Integer> list = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            list.add(i, -1);
        }

        return list;
    }

    public void printAdjLists() {
        String output = "";
        for (int m = 0; m < adjLists.size(); m++) {
            output += "{ Molecule #" + (m + 1) + ":\r\n";
            ArrayList<HashSet<Integer>> adjList = adjLists.get(m);
            for (int i = 0; i < adjList.size(); i++) {
                output += "\t" + (i + 1) + ": [";

                int j = 0;
                for (int v : adjList.get(i)) {
                    output += (v + 1);
                    output += (j++ < adjList.get(i).size() - 1) ? ", " : "";
                }
                output += "]\r\n";
            }
            output += "}\r\n";
        }
        System.out.println(output);
    }

    public void printLoops() {
        String output = "";
        for (int m = 0; m < loopAdjList.size(); m++) {
            output += "{ Molecule #" + (m + 1) + ":\r\n";
            ArrayList<HashSet<Integer>> adjList = loopAdjList.get(m);
            for (int i = 0; i < adjList.size(); i++) {
                output += "\t" + (i + 1) + ": [";

                int j = 0;
                for (Integer v : adjList.get(i)) {
                    output += (v + 1);
                    output += (j++ < adjList.get(i).size() - 1) ? ", " : "";
                }
                output += "]\r\n";
            }
            output += "}\r\n";
        }
        System.out.println(output);
    }

    public void printCenters() {
        String output = "";
        for (int i = 0; i < adjLists.size(); i++) {
            output += "{ Molecule #" + (i + 1) + ": [ ";

            int j = 0;
            for (int c : roots.get(i)) {
                output += (c + 1);
                output += (j++ < roots.get(i).length - 1) ? ", " : "";
            }
            output += " ] }\r\n";
        }
        System.out.println(output);
    }

    public void printHeights() {
        String output = "";
        for (int i = 0; i < adjLists.size(); i++) {
            output += "{ Molecule #" + (i + 1) + ": [ ";

            int j = 0;
            for (int h : heights.get(i)) {
                output += (h + 1);
                output += (j++ < heights.get(i).size() - 1) ? ", " : "";
            }
            output += " ] }\r\n";
        }
        System.out.println(output);
    }

    public boolean expect(int[] something, int[] tobe) {
        if (something.length != tobe.length) return false;

        boolean result = true;
        for (int i = 0; i < tobe.length; i++) {
            result &= something[i] == tobe[i];
        }

        return result;
    }

    public void readInputFromCLI() {
        Iterator<String> lines;
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String line = reader.readLine();
            StringTokenizer st = new StringTokenizer(line, " ");
            int A = Integer.parseInt(st.nextToken()), B = Integer.parseInt(st.nextToken()), M = Integer.parseInt(st.nextToken());

            M_edges = B;
            N_nodes = A;
            for (int m = 0; m < M; m++) {
                int b = 0;
                ArrayList<HashSet<Integer>> adjList = initializedIntegerAdjList(A);

                heights.add(initializedIntegerList(A));
                loopAdjList.add(initializedIntegerAdjList(A));
                loopParents.add(initializedBooleanList(A));
                inFIFO = new boolean[M][N_nodes];
                removed = new boolean[M][N_nodes];
                waiting = new boolean[M][N_nodes];
                removedSiblings = new int[M][N_nodes];
                while(b++ < B) {
                    line = reader.readLine();
                    st = new StringTokenizer(line, " ");
                    int v1 = Integer.parseInt(st.nextToken()), v2 = Integer.parseInt(st.nextToken());
                    adjList.get(v1 - 1).add(v2 - 1);
                    adjList.get(v2 - 1).add(v1 - 1);
                }
                adjLists.add(adjList);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void readInputFromDataset(String path) {
        Iterator<String> lines;
        try {
            lines = Files.lines(Paths.get(BASE_PATH + path)).iterator();
            String[] input = lines.next().split(" ");
            int A = Integer.parseInt(input[0]), B = Integer.parseInt(input[1]), M = Integer.parseInt(input[2]);
            System.out.println("\r\n{ A: " + A + ", B: " + B + ", M: " + M + " }");

            M_edges = B;
            N_nodes = A;
            inFIFO = new boolean[M][N_nodes];
            removed = new boolean[M][N_nodes];
            waiting = new boolean[M][N_nodes];
            removedSiblings = new int[M][N_nodes];
            for (int m = 0; m < M; m++) {
                int b = 0;
                ArrayList<HashSet<Integer>> adjList = initializedIntegerAdjList(A);

                heights.add(initializedIntegerList(A));
                loopAdjList.add(initializedIntegerAdjList(A));
                loopParents.add(initializedBooleanList(A));
                while(b++ < B) {
                    input = lines.next().split(" ");
                    int v1 = Integer.parseInt(input[0]), v2 = Integer.parseInt(input[1]);
                    adjList.get(v1 - 1).add(v2 - 1);
                    adjList.get(v2 - 1).add(v1 - 1);
                }
                adjLists.add(adjList);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int[] readOutputFromDataset(String path) {
        Iterator<String> lines;
        try {
            lines = Files.lines(Paths.get(BASE_PATH + path)).iterator();
            String[] input = lines.next().split(" ");
            int[] result = new int[input.length];
            for (int i = 0; i < input.length; i++) {
                result[i] = Integer.parseInt(input[i]);
            }
            return result;
        } catch (IOException e) {
            e.printStackTrace();
        }

        return new int[] {};
    }

    public int[] runFor(String env, String path) {
        if (!env.equals("dataset")) return new int[] {};
        readInputFromDataset(path);
        return findStructures();
    }

    public int[] runFor(String env) {
        if (!env.equals("cli")) return new int[] {};

//        long startTime = System.nanoTime();
        readInputFromCLI();
//        long stopTime = System.nanoTime();
//        System.out.println("Reading data: " + (stopTime - startTime)/ 1000000000.0);

        return findStructures();
    }

    public void identifyLoops() {
        int totalLoops = M_edges - (N_nodes - 1);
        if (totalLoops == 0) return;

        for (int i = 0; i < adjLists.size(); i++) {
            int count = 0;

            int[] checkedSiblings = new int[N_nodes];
            next: for (int j = 0; j < N_nodes; j++) {
                int amountOfSiblingsJ = adjLists.get(i).get(j).size();
                if (amountOfSiblingsJ - checkedSiblings[j] <= 0) continue;

                if (amountOfSiblingsJ == 3 && loopAdjList.get(i).get(j).isEmpty()) {
                    keepLooking: for (Integer currentSibling : adjLists.get(i).get(j)) {
                        int amountOfSiblingsCS = adjLists.get(i).get(currentSibling).size();
                        if (amountOfSiblingsCS - checkedSiblings[currentSibling] <= 0) continue;

                        checkedSiblings[j]++;
                        checkedSiblings[currentSibling]++;
                        if (loopAdjList.get(i).get(currentSibling).isEmpty()) {
                            for (Integer otherSibling : adjLists.get(i).get(j)) {
                                if (adjLists.get(i).get(currentSibling).contains(otherSibling)) {
                                    checkedSiblings[currentSibling]++;
                                    checkedSiblings[otherSibling]++;

                                    loopAdjList.get(i).get(j).add(currentSibling);
                                    loopAdjList.get(i).get(j).add(otherSibling);

                                    loopAdjList.get(i).get(currentSibling).add(j);
                                    loopAdjList.get(i).get(currentSibling).add(otherSibling);

                                    loopAdjList.get(i).get(otherSibling).add(j);
                                    loopAdjList.get(i).get(otherSibling).add(currentSibling);
                                    count++;

                                    if (count == totalLoops) break next;
                                    else break keepLooking;
                                }
                            }
                        }
                    }
                }
            }

        }
    }

    public int[] center(int m) {
        ArrayList<HashSet<Integer>> adjList = adjLists.get(m);
        ArrayList<Integer> currentHeights = heights.get(m);
        ArrayList<HashSet<Integer>> currentLoops = loopAdjList.get(m);

        ArrayDeque<Integer> leaves = new ArrayDeque<>();

        for (int i = 0; i < adjList.size(); i++) {
            if (adjList.get(i).size() <= 1) {
                removedSiblings[m][i]++;
                leaves.addLast(i);
                currentHeights.set(i, 0);
                inFIFO[m][i] = true;
            }
        }

        int totalRemoved = 0;
        while(totalRemoved++ + 2 < adjList.size()) {
            if (leaves.isEmpty()) break;

            Integer leaf = leaves.removeFirst();
            removed[m][leaf] = true;
            for (Integer s : adjList.get(leaf)) {
                removedSiblings[m][s]++;
                int oldHeight = currentHeights.get(s);
                int newHeight = currentHeights.get(leaf) + 1;

                currentHeights.set(s, removed[m][s] || oldHeight > newHeight ? oldHeight : newHeight);

                if (!inFIFO[m][s] && adjList.get(s).size() - removedSiblings[m][s] == 1) {
                    // natural leaves
                    leaves.addLast(s);
                    inFIFO[m][s] = true;
                } else if (!inFIFO[m][s] && adjList.get(s).size() - removedSiblings[m][s] == 2 && !waiting[m][s]) {
                    // loop leaves
                    if (currentLoops.get(s).size() > 0) {
                        leaves.addLast(s);
                        inFIFO[m][s] = true;
                        for (Integer sibling : currentLoops.get(s)) waiting[m][sibling] = true;
                    }
                }
            }

        }

        Integer v1 = leaves.removeLast(), v2 = leaves.removeFirst();
        Integer h1 = currentHeights.get(v1), h2 = currentHeights.get(v2);
        return Objects.equals(h1, h2) ? new int[] {v1, v2} : (h1 > h2 ? new int[] {v1} : new int[] {v2});
    }

    public void findRoots() {
        for (int i = 0; i < adjLists.size(); i++) {
            roots.add(center(i));
        }
    }

    public String treeCoding(int v1, boolean[] visited, String output, int skip, int molecule) {
        if (visited[v1]) return output;

        ArrayList<HashSet<Integer>> adjList = adjLists.get(molecule);
        ArrayList<HashSet<Integer>> loops = loopAdjList.get(molecule);
        ArrayList<Boolean> parents = loopParents.get(molecule);

        boolean isLoopParent = parents.get(v1);
        visited[v1] = true;
        ArrayList<String> all = new ArrayList<>();

        for (Integer v2: adjList.get(v1)) {
            boolean hasLoop = loops.get(v2).size() > 0;
            boolean isChildLoopParent = parents.get(v2);

            if (!visited[v2] && v2 != skip) {
                if(!hasLoop || isLoopParent || isChildLoopParent)
                    all.add(0 + treeCoding(v2, visited, "", skip, molecule) + 1);
            }
        }

        Collections.sort(all);
        output = String.join("", all);

        // Check if closing parent of loop
        if (isLoopParent) output += 2;

        return output;
    }

    public int[] computeFinalResult() {
        int[] result = new int[isomorphisms.size()];

        int i = 0;
        for (String key : isomorphisms.keySet()) result[i++] = isomorphisms.get(key);
        Arrays.sort(result);

        return result;
    }

    public void setLoopParents() {
        for (int m = 0; m < adjLists.size(); m++) {
            ArrayList<HashSet<Integer>> currentLoops = loopAdjList.get(m);
            ArrayList<Integer> currentHeights = heights.get(m);
            for (int v1 = 0; v1 < adjLists.get(m).size(); v1++) {
                if (currentLoops.get(v1).size() > 0) {
                    Iterator<Integer> it = currentLoops.get(v1).iterator();
                    int v2 = it.next();
                    int v3 = it.next();

                    int h1 = currentHeights.get(v1);
                    int h2 = currentHeights.get(v2);
                    int h3 = currentHeights.get(v3);

                    loopParents.get(m).set(h1 > h2 ? (h1 > h3 ? v1 : v3) : (h2 > h3 ? v2 : v3), true);
                }
            }
        }
    }

    public String computeCertificate(int molecule) {
        int[] moleculeRoots = roots.get(molecule);
        if (moleculeRoots.length == 2) {
            String codeA = 0 + treeCoding(moleculeRoots[0], new boolean[N_nodes], "", moleculeRoots[1], molecule) + 1;
            String codeB = 0 + treeCoding(moleculeRoots[1], new boolean[N_nodes], "", moleculeRoots[0], molecule) + 1;

            return codeA.compareTo(codeB) <= 0 ? codeA + codeB : codeB + codeA;
        }

        return treeCoding(moleculeRoots[0], new boolean[N_nodes], "", -1, molecule);
    }

    public void computeCertificates() {
        for (int i = 0; i < adjLists.size(); i++) {
            String certificate = computeCertificate(i);
            isomorphisms.put(certificate, isomorphisms.containsKey(certificate) ? isomorphisms.get(certificate) + 1 : 1);
        }
    }

    public int[] findStructures() {
//        long startTime = System.nanoTime();
        identifyLoops();
//        long stopTime = System.nanoTime();
//        System.out.println("Identifying loops     : " + (stopTime - startTime)/ 1000000000.0);
//        printLoops();

//        startTime = System.nanoTime();
        findRoots();
//        stopTime = System.nanoTime();
//        System.out.println("Finding roots         : " + (stopTime - startTime)/ 1000000000.0);

//        startTime = System.nanoTime();
        setLoopParents();
//        stopTime = System.nanoTime();
//        System.out.println("Setting loop parents  : " + (stopTime - startTime)/ 1000000000.0);

//        startTime = System.nanoTime();
        computeCertificates();
        int[] r = computeFinalResult();
//        stopTime = System.nanoTime();
//        System.out.println("Computing certificates: " + (stopTime - startTime)/ 1000000000.0);

        return r;
    }

    public static void test() {
        for (int i = 10; i <= 10; i++) {
            long startTime = System.nanoTime();

            Main s = new Main();
            String filename = "pub" + (i >= 10 ? i : "0" + i);
            int[] result = s.runFor("dataset", filename + ".in");
            int[] expected = s.readOutputFromDataset(filename + ".out");

            long stopTime = System.nanoTime();
            System.out.println("total time: " + (stopTime - startTime)/ 1000000000.0);
            String checkMark = s.expect(result, expected)  ? "✔" : "✘";
            System.out.println(checkMark + " - {expected: " + Arrays.toString(expected) + ", result:" + Arrays.toString(result) + " };\r\n");
        }
    }

    public static void conquer() {
        Main s = new Main();
        System.out.println(String.join(" ", Arrays.toString((s.runFor("cli"))).split("[\\[\\]]")[1].split(", ")));
    }

    public static void main(String[] args) {
//        Main.test();
        Main.conquer();
    }
}
