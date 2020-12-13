package pal;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Main {
    private static String BASE_PATH = "/Users/ananias/Documents/CTU/Advanced Algorithms/Labs/05/dataset/";
    private int N_total = 0;
    private int M_wagons = 0;
    private int K_changes = 0;
    private String reference = "";
    private String sequence = "";
    private HashMap<String, int[]> subsequencesLevenshtein = new HashMap<>();
    private int[][] subsequencesMatrix;
    private long[] memoMax;
    private long[] memoMin;

    public static String center(String text, int len){
        if (len <= text.length())
            return text.substring(0, len);
        int before = (len - text.length())/2;
        if (before == 0)
            return String.format("%-" + len + "s", text);
        int rest = len - before;
        return String.format("%" + before + "s%-" + rest + "s", "", text);
    }

    public static void printLevenshteinMatrix(String reference, String sequence, ArrayList<ArrayList<Integer>> levenshtein) {
        int n = Math.max(reference.length(), sequence.length());
        String output = "\r\n+" + String.join("", Collections.nCopies(sequence.length() + 2, String.join("", Collections.nCopies(n, "-")) + "+")) + "\r\n|";
        output += center("START", n) + "|";
        output += center(" ", n) + "|";
        for (char c: sequence.toCharArray()) {
            output += center(Character.toString(c), n) + "|";
        }
        output += "\r\n|" + center(" ", n) + "|";
        output += center(Integer.toString(0), n) + "|";

        for (int i = 0; i < sequence.toCharArray().length; i++) {
            output += center(Integer.toString(i + 1), n) + "|";
        }
        output += "\r\n";

        int i = 0;
        for (ArrayList<Integer> row: levenshtein) {
            if (i == 0) {
                i++;
                continue;
            }
            output += "|" + center(Character.toString(reference.charAt(i - 1)), n) + "|";
            output += center(Integer.toString(i), n) + "|";
            int j = 0;
            for (int value: row) {
                if (j == 0) {
                    j++;
                    continue;
                }
                output += center(Integer.toString(value), n) + "|";
                j++;
            }
            output += "\r\n";
            i++;
        }
        System.out.println(output);
    }

    public void printAllLevSubsequences() {
        ArrayList<String> subsequences = new ArrayList<>(subsequencesLevenshtein.keySet());
        Collections.sort(subsequences, new Comparator<String>(){
            @Override
            public int compare (String a, String b) {
                int diff = a.length() - b.length();
                return diff == 0 ? a.compareTo(b) : diff;
            }
        });
        String output = "";
        for (String subsequence: subsequences) {
            int last = subsequencesLevenshtein.get(subsequence).length - 1;
            if (subsequencesLevenshtein.get(subsequence)[last] <= K_changes) {
                output += "{ " + subsequence + ": " + subsequencesLevenshtein.get(subsequence)[last] + " }\r\n";
            }
        }
        System.out.println(output);
    }

    public void readInputFromDataset(String path) {
        Iterator<String> lines;
        try {
            lines = Files.lines(Paths.get(BASE_PATH + path)).iterator();
            sequence = lines.next();
//            sequence = String.join("", Collections.nCopies(7000, "d"));
            reference = lines.next();
//            reference = String.join("", Collections.nCopies(140, "d"));
            K_changes = Integer.parseInt(lines.next());
            N_total = sequence.length();
            M_wagons = reference.length();
            subsequencesMatrix = new int[N_total][N_total];
            for (int[] row : subsequencesMatrix) Arrays.fill(row, -1);


            memoMax = new long[N_total];
            Arrays.fill(memoMax, -1);
            memoMin = new long[N_total];
            Arrays.fill(memoMin, -1);

            System.out.println("\r\n{ N: " + N_total + ", M: " + M_wagons + ", K: " + K_changes + " }");
            System.out.println(sequence);
            System.out.println(reference);
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

    public void readInputFromCLI() {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            sequence = reader.readLine();
            reference = reader.readLine();
            K_changes = Integer.parseInt(reader.readLine());
            N_total = sequence.length();
            M_wagons = reference.length();
            subsequencesMatrix = new int[N_total][N_total];
            for (int[] row : subsequencesMatrix) Arrays.fill(row, -1);

            memoMax = new long[N_total];
            Arrays.fill(memoMax, -1);
            memoMin = new long[N_total];
            Arrays.fill(memoMin, -1);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public int[] runFor(String env, String path) {
        if (!env.equals("dataset")) return new int[] {};
        readInputFromDataset(path);

        return calculateTrainArrays();
    }

    public int[] runFor(String env) {
        if (!env.equals("cli")) return new int[] {};

//        long startTime = System.nanoTime();
        readInputFromCLI();
//        long stopTime = System.nanoTime();
//        System.out.println("Reading data: " + (stopTime - startTime)/ 1000000000.0);

        return calculateTrainArrays();
    }

    public int computeLevenshtein(String a, String b) {
        int[][] levenshtein = new int[a.length() + 1][b.length() + 1];
        for (int i = 0; i <= a.length(); i++) {
            levenshtein[i][0] = i;
        }

        for (int j = 1; j <= b.length(); j++) {
            Character bChar = b.charAt(j - 1);
            levenshtein[0][j] = j;

            for (int i = 1; i <= a.length(); i++) {
                int diag = levenshtein[i - 1][j - 1];
                int cost = a.charAt(i - 1) == bChar ? 0 : 1;
                levenshtein[i][j] = Math.min(Math.min(levenshtein[i - 1][j] + 1, levenshtein[i][j - 1] + 1), diag + cost);
            }
        }

        return levenshtein[a.length()][b.length()];
    }

    public int computeLevenshteinFromSubsequence(int start, int window) {
        int end = start + window;
        int[] oldLev = subsequencesLevenshtein.get(sequence.substring(start, end - 1));
        int[] levenshtein = new int[reference.length() + 1];

        char bChar = sequence.charAt(end - 1);
        levenshtein[0] = window;
        for (int i = 1; i <= reference.length(); i++) {
            int diag = oldLev[i - 1];
            int cost = reference.charAt(i - 1) == bChar ? 0 : 1;
            levenshtein[i] = Math.min(Math.min(oldLev[i] + 1, levenshtein[i - 1] + 1), diag + cost);
        }

        subsequencesLevenshtein.put(sequence.substring(start, end), levenshtein);
        return subsequencesMatrix[start][end - 1] = levenshtein[reference.length()];
    }
//
    public void initializeSubsequenceMap() {
        subsequencesLevenshtein.put("", new int[reference.length() + 1]);
        subsequencesLevenshtein.get("")[0] = 0;
        for (int i = 1; i <= reference.length(); i++) subsequencesLevenshtein.get("")[i] = i;
    }

    public long dpMin(int pos) {
        if(pos == -1) return 0;
        if(memoMin[pos] != -1) return memoMin[pos];

        long best = Integer.MAX_VALUE;
        for(int i = 0; i <= pos; i++) {
            int lev = subsequencesMatrix[i][pos];
            if (lev > -1 && lev <= K_changes) best = Math.min(best, dpMin(i - 1));
        }

        return memoMin[pos] = best + 1;
    }

    public long dpMax(int pos) {
        if(pos == -1) return 0;
        if(memoMax[pos] != -1) return memoMax[pos];

        long best = Integer.MIN_VALUE;
        for(int i = 0; i <= pos; i++) {
            int lev = subsequencesMatrix[i][pos];
            if (lev > -1 && lev <= K_changes) best = Math.max(best, dpMax(i - 1));
        }

        return memoMax[pos] = best + 1;
    }

    public int[] calculateTrainArrays() {
        int count = 0;
        int total = 0;
        // Could implement smarter init, considering window min size
        initializeSubsequenceMap();
        for (int window = 1; window <= M_wagons + K_changes; window++) {
            for (int index = 0; index + window  <= sequence.length(); index++) {
                String sub = sequence.substring(index, index + window);
                computeLevenshteinFromSubsequence(index, window);
                count++;
                total += reference.length() * (window);
            }
        }
//        printAllLevSubsequences();
//        System.out.println(count + " levenshtein computations for a total of " + total + " operations");
        return new int[] {(int) dpMin(sequence.length() - 1), (int) dpMax(sequence.length() - 1)};
//        return new int[] {0, 0};
    }

    public static void test() {
        for (int i = 1; i <= 10; i++) {
            long startTime = System.nanoTime();

            Main s = new Main();
            String filename = "pub" + (i >= 10 ? i : "0" + i);
            int[] result = s.runFor("dataset", filename + ".in");
            String[] expected = s.readOutputFromDataset(filename + ".out").split(" ");

            long stopTime = System.nanoTime();
            System.out.println("total time: " + (stopTime - startTime)/ 1000000000.0);
            String checkMark = result[0] == Integer.parseInt(expected[0]) && result[1] == Integer.parseInt(expected[1])  ? "✔" : "✘";
            System.out.println(checkMark + " - { expected: [ " + expected[0] + " " + expected[1] + " ], result: [ " + result[0] + " " + result[1] + " ] };\r\n");
        }
    }

    public static void conquer() {
        Main s = new Main();
        int[] result = s.runFor("cli");
        System.out.println(result[0] + " " + result[1]);
    }

    public static void main(String[] args) {
//        Main.test();
        Main.conquer();
    }
}
