package hilarjos;

import java.util.*;

public class Print {
    public static void InformationSets(ArrayList<HashSet<String>> agentInformationSets, ArrayList<HashMap<String, HashSet<String>>> banditInformationSets) {
        int i = 0;
        System.out.println("Agent:");
        for (HashSet<String> actionsByLevel: agentInformationSets) {
            int j = 0;
            String output = "[ ";
            output += Arrays.toString(actionsByLevel.toArray()) + ", ";
            output = output.substring(0, output.length() - 2) + " ]";
            System.out.println("Level " + ++i + ": " + (i < 10 ? " " : "") + output);
        }

        i = 0;
        System.out.println("ATTACKER:");
        for (HashMap<String, HashSet<String>> actionsByLevel: banditInformationSets) {
            int j = 0;
            String output = "[ ";
            for (String key: actionsByLevel.keySet()) {
                if (actionsByLevel.get(key).contains("Fail")) continue;
                output += Arrays.toString(actionsByLevel.get(key).toArray()) + ", ";
            }
            if (output.length() == 2) continue;

            output = output.substring(0, output.length() - 2) + " ]";
            System.out.println("Level " + ++i + ": " + (i < 10 ? " " : "") + output);
        }
    }

    public static void RealizationPlans(HashMap<String, HashMap<String, ArrayList<String>>> banditUtilities, HashMap<String, HashMap<String, ArrayList<String>>> agentUtilities) {
        int i = 0;
        System.out.println("SOLUTION_AGENT:");
        for (String keyInformationSet: banditUtilities.keySet()) {
            for (String keyActionTaken: banditUtilities.get(keyInformationSet).keySet()) {
                System.out.println("S" + ++i + ": " + (i < 10 ? " " : "") + "{ " + keyActionTaken + ": v[" + keyInformationSet + "] ≤ " + String.join(" + ", banditUtilities.get(keyInformationSet).get(keyActionTaken)) + " }");
            }
        }

        i = 0;
        System.out.println("SOLUTION_ATTACKER:");
        for (String keyInformationSet: agentUtilities.keySet()) {
            for (String keyActionTaken: agentUtilities.get(keyInformationSet).keySet()) {
                System.out.println("Q" + ++i + ": " + (i < 10 ? " " : "") + "{ " + keyActionTaken + ": v[" + keyInformationSet + "] ≥ " + String.join(" + ", agentUtilities.get(keyInformationSet).get(keyActionTaken)) + " }");
            }
        }
    }

    public static void Map(String[][] map) {
        for (String[] row: map) {
            for (String element: row) {
                if (element.contains("E")) System.out.print("E");
                else if (element.contains("G")) System.out.print("G");
                else System.out.print(element);
            }
            System.out.print("\r\n");
        }
    }

    public static void NFCs(HashMap<String, HashSet<String>> agentNFCs, HashMap<String, HashMap<String, HashSet<String>>> banditsNFCs) {
        System.out.println("\r\nNFCs");

        Comparator<String> cmp = new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                int diff = a.length() - b.length();
                return diff == 0 ? a.compareTo(b) : diff;
            }
        };

        List<String> sortedKeys = Arrays.asList(agentNFCs.keySet().toArray(new String[agentNFCs.size()]));
        Collections.sort(sortedKeys, cmp);
        System.out.println("AGENT");
        for (String key: sortedKeys) {
            String output = "";
            if (key.matches("^-?\\d+$")) output += String.join(" + ", agentNFCs.get(key)) + " = " + key;
            else output += key + " = " + String.join(" + ", agentNFCs.get(key));
            System.out.println(output);
        }

        sortedKeys = Arrays.asList(banditsNFCs.keySet().toArray(new String[banditsNFCs.size()]));
        Collections.sort(sortedKeys, cmp);
        System.out.println("ATTACKER");
        for (String key: sortedKeys) {
            HashSet<String> outputSet = new HashSet<>();
            for (String alarmPosition: banditsNFCs.get(key).keySet()) {
                String output = " [" + alarmPosition + "]: ";
                if (key.matches("^-?\\d+$")) output += String.join(" + ", banditsNFCs.get(key).get(alarmPosition)) + " = " + key + "   ";
                else output += key + " = " + String.join(" + ", banditsNFCs.get(key).get(alarmPosition)) + "   ";
                outputSet.add(output);
            }
            System.out.println(outputSet);
        }
    }

    public static void Sequences(ArrayList<ArrayList<String>> agentActions, ArrayList<ArrayList<String>> banditsNoNature) {
        int i = 0;
        System.out.println("AGENT:");
        for (ArrayList<String> sequence: agentActions) {
            System.out.println("S" + ++i + ": " + (i < 10 ? " " : "") + Arrays.toString(sequence.toArray()));
        }

        i = 0;
        System.out.println("ATTACKER:");
        for (ArrayList<String> sequence: banditsNoNature) {
            System.out.println("Q" + ++i + ": " + (i < 10 ? " " : "") + Arrays.toString(sequence.toArray()));
        }
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

    public static void ExtendedUtilityFunctions(ArrayList<ArrayList<String>> agentActions,
                                              ArrayList<ArrayList<String>> banditsNoNature,
                                              HashMap<String, HashMap<String, Double>> utilities) {
        String header = "|" + Print.center("AGENT\\ATTACKER", 40) + "|";
        for (ArrayList<String> si: agentActions) {
            header += Print.center(Arrays.toString(si.toArray()), 40) + "|";
//            header += Print.center("S" + (i + 1), 60) + "|";
        }
        System.out.println(String.join("", Collections.nCopies(header.length(), "-")));
        System.out.println(header);

        int j = 0;
        for (ArrayList<String> qj: banditsNoNature) {
            String line1 = "|" + Print.center("",40) + "|";
            String line2 = "|" + Print.center(Arrays.toString(qj.toArray()),40) + "|";
            String line3 = "|" + Print.center("",40) + "|";

            int i = 0;
            for (ArrayList<String> si : agentActions) {
                String e = "";
                String sKey = Arrays.toString(si.toArray()), qKey = Arrays.toString(qj.toArray());
                e += utilities.get(sKey).containsKey(qKey) ? utilities.get(sKey).get(qKey) : "0.0";
                // utility function
                line1 += Print.center("", 40) + "|";
                line2 += Print.center(e, 40) + "|";
                line3 += Print.center("", 40) + "|";

                i++;
            }
            System.out.println(String.join("", Collections.nCopies(line1.length(), "-")));
            System.out.println(line1);
            System.out.println(line2);
            System.out.println(line3);

            j++;
        }
        System.out.println(String.join("", Collections.nCopies(header.length(), "-")));
    }

    public static void Game(Main s) {
        Print.Map(s.getMap());
        Print.Sequences(s.getAgentActions(), s.getBanditsNoNature());
//        Print.NFCs(s.getAgentNFCs(), s.getBanditsNFCs());
//        Print.InformationSets(s.getAgentInformationSets(), s.getBanditsInformationSets());
        Print.ExtendedUtilityFunctions(s.getAgentActions(), s.getBanditsNoNature(), s.getUtilities());
        Print.RealizationPlans(s.getBanditUtilities(), s.getAgentUtilities());
    }
}
