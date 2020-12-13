package pal;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Main {
    private static String BASE_PATH = "/Users/ananias/Documents/CTU/Advanced Algorithms/Labs/04/dataset/";
    private HashMap<String, Integer> L = new HashMap<>();
    private ArrayList<String> mapL = new ArrayList<>(10);
    private int N_A;
    private int M_A;
    private HashSet<Integer> finalsA = new HashSet<>();
    private int N_B;
    private int M_B;
    private HashSet<Integer> finalsB = new HashSet<>();
    private ArrayList<ArrayList<ArrayList<Integer>>> adjMatrixA = new ArrayList<>(1000000);
    private ArrayList<ArrayList<ArrayList<Integer>>> adjMatrixB = new ArrayList<>(1000000);
    private ArrayList<ArrayList<Integer>> jointVisited = new ArrayList<>(1000000);

    public ArrayList<ArrayList<ArrayList<Integer>>> initializedAdjMatrix(int n, int m) {
        ArrayList<ArrayList<ArrayList<Integer>>> adjMatrix = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            ArrayList<ArrayList<Integer>> states = new ArrayList<>(m);
            for (int j = 0; j < m; j++) states.add(j, new ArrayList<>());
            adjMatrix.add(i, states);
        }

        return adjMatrix;
    }

    public ArrayList<ArrayList<Integer>> initializedVisitedMatrix(int n, int m) {
        ArrayList<ArrayList<Integer>> adjMatrix = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            ArrayList<Integer> visited = new ArrayList<>(m);
            for (int j = 0; j < m; j++) visited.add(j, Integer.MAX_VALUE);
            adjMatrix.add(i, visited);
        }

        return adjMatrix;
    }

    public static String center(String text, int len){
        if (len <= text.length())
            return text.substring(0, len);
        int before = (len - text.length())/2;
        if (before == 0)
            return String.format("%-" + len + "s", text);
        int rest = len - before;
        return String.format("%" + before + "s%-" + rest + "s", "", text);
    }

    public void printAdjMatrices() {
        String output = "";
        for (int i = 0; i < adjMatrixA.size(); i++) {
            output += i + ": [";
            for (int j = 0; j < adjMatrixA.get(i).size(); j++) {
                output += center(Arrays.toString(adjMatrixA.get(i).get(j).toArray()),20);
            }
            output = output.substring(0, output.length() - 2);
            output += "]\r\n";
        }
        System.out.print(output);
        output = "Finals: [ ";
        for (Integer finalA: finalsA) {
            output += finalA + ", ";
        }
        output = output.substring(0, output.length() - 2);
        output += " ]\r\n";
        System.out.println(output);


        output = "";
        for (int i = 0; i < adjMatrixB.size(); i++) {
            output += i + ": [";
            for (int j = 0; j < adjMatrixB.get(i).size(); j++) {
                output += center(Arrays.toString(adjMatrixB.get(i).get(j).toArray()),20);
            }
            output = output.substring(0, output.length() - 2);
            output += "]\r\n";
        }
        System.out.print(output);
        output = "Finals: [ ";
        for (Integer finalB: finalsB) {
            output += finalB + ", ";
        }
        output = output.substring(0, output.length() - 2);
        output += " ]\r\n";
        System.out.println(output);
    }

    public void readInputFromDataset(String path) {
        Iterator<String> lines;
        try {
            lines = Files.lines(Paths.get(BASE_PATH + path)).iterator();
            char[] charArr = lines.next().toCharArray();
            Arrays.sort(charArr);
            String alphabet = Arrays.toString(charArr);
            alphabet = alphabet.substring(1, alphabet.length() - 1);
            alphabet = alphabet.replaceAll(", ", "");

            for (int i = 0; i < alphabet.length(); i++) {
                L.put(Character.toString(alphabet.charAt(i)), i);
                mapL.add(Character.toString(alphabet.charAt(i)));
            }
            String[] input = lines.next().split(" ");
            N_A = Integer.parseInt(input[0]);
            M_A = Integer.parseInt(input[1]);

            adjMatrixA = initializedAdjMatrix(N_A, L.keySet().size());
            for (int m = 0; m < M_A; m++) {
                input = lines.next().split(" ");
                int v1 = Integer.parseInt(input[0]), v2 = Integer.parseInt(input[2]);
                String c = input[1];

                adjMatrixA.get(v1).get(L.get(c)).add(v2);
            }
            String[] allFinal = lines.next().split(" ");
            int i = 0;
            for (String state: allFinal) finalsA.add(Integer.parseInt(state));

        /*  --------------------------------------------------------------------------------------------------------  */
            input = lines.next().split(" ");
            N_B = Integer.parseInt(input[0]);
            M_B = Integer.parseInt(input[1]);

            adjMatrixB = initializedAdjMatrix(N_B, L.keySet().size());
            for (int m = 0; m < M_B; m++) {
                input = lines.next().split(" ");
                int v1 = Integer.parseInt(input[0]), v2 = Integer.parseInt(input[2]);
                String c = input[1];

                adjMatrixB.get(v1).get(L.get(c)).add(v2);
            }

            allFinal = lines.next().split(" ");
            for (String state: allFinal) finalsB.add(Integer.parseInt(state));

        /*  --------------------------------------------------------------------------------------------------------  */
            jointVisited = initializedVisitedMatrix(N_A, N_B);
            System.out.println("\r\n{ L: " + Arrays.toString(L.keySet().toArray()) + ", A: [ N: " + N_A + ", M: " + M_A + " ], B: [ N: " + N_B + ", M: " + M_B + " ] }");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String readOutputFromDataset(String path) {
        Iterator<String> lines;
        try {
            lines = Files.lines(Paths.get(BASE_PATH + path)).iterator();
            return lines.next();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return "";
    }

    public String runFor(String env, String path) {
        if (!env.equals("dataset")) return "";
        readInputFromDataset(path);
//        return findStructures();

        return budWord();
    }

    public void readInputFromCLI() {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            char[] charArr = reader.readLine().toCharArray();
            Arrays.sort(charArr);
            String alphabet = Arrays.toString(charArr);
            alphabet = alphabet.substring(1, alphabet.length() - 1);
            alphabet = alphabet.replaceAll(", ", "");

            for (int i = 0; i < alphabet.length(); i++) {
                L.put(Character.toString(alphabet.charAt(i)), i);
                mapL.add(Character.toString(alphabet.charAt(i)));
            }
            String line = reader.readLine();
            StringTokenizer st = new StringTokenizer(line, " ");
            N_A = Integer.parseInt(st.nextToken());
            M_A = Integer.parseInt(st.nextToken());

            adjMatrixA = initializedAdjMatrix(N_A, L.keySet().size());
            for (int m = 0; m < M_A; m++) {
                line = reader.readLine();
                st = new StringTokenizer(line, " ");
                int v1 = Integer.parseInt(st.nextToken());
                String c = st.nextToken();
                int v2 = Integer.parseInt(st.nextToken());

                adjMatrixA.get(v1).get(L.get(c)).add(v2);
            }
            line = reader.readLine();
            st = new StringTokenizer(line, " ");
            finalsA = new HashSet<>();
            while (st.hasMoreTokens()) finalsA.add(Integer.parseInt(st.nextToken()));

        /*  --------------------------------------------------------------------------------------------------------  */
            line = reader.readLine();
            st = new StringTokenizer(line, " ");
            N_B = Integer.parseInt(st.nextToken());
            M_B = Integer.parseInt(st.nextToken());

            adjMatrixB = initializedAdjMatrix(N_B, L.keySet().size());
            for (int m = 0; m < M_B; m++) {
                line = reader.readLine();
                st = new StringTokenizer(line, " ");
                int v1 = Integer.parseInt(st.nextToken());
                String c = st.nextToken();
                int v2 = Integer.parseInt(st.nextToken());

                adjMatrixB.get(v1).get(L.get(c)).add(v2);
            }
            line = reader.readLine();
            st = new StringTokenizer(line, " ");
            finalsB = new HashSet<>();
            while (st.hasMoreTokens()) finalsB.add(Integer.parseInt(st.nextToken()));

        /*  --------------------------------------------------------------------------------------------------------  */
            jointVisited = initializedVisitedMatrix(N_A, N_B);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String runFor(String env) {
        if (!env.equals("cli")) return "";

//        long startTime = System.nanoTime();
        readInputFromCLI();
//        long stopTime = System.nanoTime();
//        System.out.println("Reading data: " + (stopTime - startTime)/ 1000000000.0);

        return budWord();
    }

    public int[] persistTrueMin(int candidateW, int[] parent, int[] joint, HashMap<String, int[]> meta) {
        ArrayList<String> results = new ArrayList<>();
        ArrayList<int[]> resultsMap = new ArrayList<>();
        ArrayList<Integer> proxy = new ArrayList<>();
        int i = 0;
        for (int w = 0; w < L.size(); w++) {
            int[] current = new int[] {joint[0], joint[1], w, joint[3]};
            String currentKey = Arrays.toString(current);

            if (meta.containsKey(currentKey)) {
                results.add(buildPath(current, meta));
                resultsMap.add(meta.get(currentKey));
                proxy.add(i++);
            }
        }

        results.add(buildPath(parent, meta) + candidateW);
        resultsMap.add(parent);
        proxy.add(i++);

        Collections.sort(proxy, new Comparator<Integer>() {
            public int compare(Integer x, Integer y) {
                String a = results.get(x), b = results.get(y);
                int diff = a.length() - b.length();

                return diff == 0 ? a.compareTo(b) : (diff > 0 ? 1 : -1);
            }
        });

        return resultsMap.isEmpty() ? new int[] {} : resultsMap.get(proxy.get(0));
    }

    public ArrayList<int[]> getSuccessors(int[] joint, HashMap<String, int[]> meta) {
        ArrayList<int[]> result = new ArrayList<>();
        for (int w = 0; w < L.size(); w++) {
            ArrayList<Integer> successorsA = adjMatrixA.get(joint[0]).get(w);
            ArrayList<Integer> successorsB = adjMatrixB.get(joint[1]).get(w);

            for (Integer sA: successorsA) {
                for (Integer sB: successorsB) {
                    int newLevel = joint[3] + 1;
                    int[] pair = new int[] {sA, sB, w, newLevel};

                    if (newLevel < jointVisited.get(sA).get(sB)) {
                        jointVisited.get(sA).set(sB, newLevel);
                        meta.put(Arrays.toString(pair), joint);
                        result.add(pair);
                    } else if (newLevel == jointVisited.get(sA).get(sB) && finalsA.contains(sA) && finalsB.contains(sB)) {
                        int[] trueParent = persistTrueMin(w, joint, pair, meta);

                        if (Arrays.toString(trueParent).equals(Arrays.toString(joint))) {
                            result.add(pair);
                            meta.put(Arrays.toString(pair), trueParent);
                        }
                    }
                }
            }
        }

        return result;
    }

    public String buildPath(int[] state, HashMap<String, int[]> steps) {
        String result = Integer.toString(state[2]);
        String stateCode = Arrays.toString(state);
        int action = steps.get(stateCode)[2];
        while (action != -1) {
            result = action + result;

            int[] step = steps.get(stateCode);
            stateCode = Arrays.toString(step);
            action = steps.get(stateCode)[2];
        }

        return result;
    }

    public String traverseBFS(int stateA, int stateB) {
        LinkedList<int[]> open = new LinkedList<>();
        HashSet<String> closed = new HashSet<>();
        HashSet<String> inOpen = new HashSet<>();
        int[] parentActionAndLevel;
        HashMap<String, int[]> meta = new HashMap<>();

        int[] startAndAction = new int[]{stateA, stateB, -1, 0};
        meta.put(Arrays.toString(startAndAction), startAndAction);
        jointVisited.get(stateA).set(stateB, 0);

        open.addLast(startAndAction);
        int solutionLevel = Integer.MAX_VALUE;
        boolean solutionFound = false;
        ArrayList<String> solutions = new ArrayList<>();
        while (!open.isEmpty()) {
            parentActionAndLevel = open.removeFirst();
            int[] parentAndAction = new int[]{parentActionAndLevel[0], parentActionAndLevel[1], parentActionAndLevel[2]};

//            System.out.println(Arrays.toString(parentActionAndLevel));
            if (solutionFound) {
                if (finalsA.contains(parentActionAndLevel[0]) && finalsB.contains(parentActionAndLevel[1]) &&
                        parentActionAndLevel[3] <= solutionLevel) {
                    solutionLevel = parentActionAndLevel[3];
                    solutions.add(buildPath(parentActionAndLevel, meta));
                }
            } else {
                if (finalsA.contains(parentActionAndLevel[0]) && finalsB.contains(parentActionAndLevel[1])) {
                    solutionLevel = parentActionAndLevel[3];
                    solutionFound = true;
                    solutions.add(buildPath(parentActionAndLevel, meta));
                }

                for (int[] childActionAndLevel : getSuccessors(parentActionAndLevel, meta)) {
                    if (!closed.contains(Arrays.toString(childActionAndLevel)) && !inOpen.contains(childActionAndLevel.toString())) {
                        String key = Arrays.toString(childActionAndLevel);
                        open.addLast(childActionAndLevel);
                        inOpen.add(childActionAndLevel.toString());
                    }
                }
                closed.add(Arrays.toString(parentAndAction));
            }
        }
//        System.out.println(Arrays.toString(solutions.toArray()));

        Collections.sort(solutions, new Comparator<String>() {
            public int compare(String a, String b) {
                int diff = a.length() - b.length();

                return diff == 0 ? a.compareTo(b) : (diff > 0 ? 1 : -1);
            }
        });
//        System.out.println(Arrays.toString(solutions.toArray()));
        return solutions.isEmpty() ? "" : solutions.get(0);
    }

    public String budWord() {
//        String rInt = traverseDFS(0, 0, "");
        String rInt = traverseBFS(0, 0);
        String result = "";
        for (int i = 0; i < rInt.length(); i++) result += mapL.get(Integer.parseInt(Character.toString(rInt.charAt(i))));

        return result;
    }

    public static void test() {
        for (int i = 1; i <= 11; i++) {
            long startTime = System.nanoTime();

            Main s = new Main();
            String filename = "pub" + (i >= 10 ? i : "0" + i);
            String result = s.runFor("dataset", filename + ".in");
            String expected = s.readOutputFromDataset(filename + ".out");
            System.out.println(result);

            long stopTime = System.nanoTime();
            System.out.println("total time: " + (stopTime - startTime)/ 1000000000.0);
            String checkMark = result.equals(expected)  ? "✔" : "✘";
            System.out.println(checkMark + " - { expected: [ " + expected + " ], result: [ " + result + " ] };\r\n");
        }
    }

    public static void conquer() {
        Main s = new Main();
        System.out.println(s.runFor("cli"));
    }

    public static void main(String[] args) {
//        Main.test();
        Main.conquer();
    }
}
