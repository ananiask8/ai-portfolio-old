package pal;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {
    private long[][] edges; // [nodeA, nodeB, weight, group]
    private int[] ranks;
    private int[] parents;
    private long[][] edgeGroups; // [start, length, weight]
    private int selectedGroupEdgeCount;
    private long mst;

    public void buildGraphFromCLI() {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String line = reader.readLine();
            StringTokenizer st = new StringTokenizer(line, " ");
            int n = Integer.parseInt(st.nextToken()), m = Integer.parseInt(st.nextToken());

            this.ranks = new int[n];
            this.parents = new int[n];
            Arrays.fill(this.parents, -1);
            this.edges = new long[m][3];
            this.selectedGroupEdgeCount = 0;
            this.mst = 0;

            for (int i = 0; i < m; i++) {
                line = reader.readLine();
                st = new StringTokenizer(line, " ");
                int v1 = Integer.parseInt(st.nextToken()) - 1, v2 = Integer.parseInt(st.nextToken()) - 1, w = Integer.parseInt(st.nextToken());
                long[] currentEdge = {v1, v2, w};
                this.edges[i] = currentEdge;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.process();
    }

    public void runForCLI() {
        this.buildGraphFromCLI();
        long mstMaxEqe = Long.MAX_VALUE;
        long includedEdges = -1;
        if (this.edgeGroups.length == 0) {
            mstMaxEqe = this.runMST(mstMaxEqe);
        } else {
            for (long[] eg : this.edgeGroups) {
                if (eg[1] < includedEdges) break;
                this.clearUF();
                this.runMSTForSelected(eg[0], eg[1]);
                if (includedEdges <= this.selectedGroupEdgeCount) {
                    includedEdges = this.selectedGroupEdgeCount;
                    long currentMST = this.runMST(mstMaxEqe);
                    mstMaxEqe = mstMaxEqe > currentMST ? currentMST : mstMaxEqe;
                }
            }
        }
        System.out.println(mstMaxEqe);
    }

    private void sortEdges() {
        Arrays.sort(this.edges, Comparator.comparingLong(a -> a[2]));
    }

    public void setEdgeGroups() {
        long start = 0, length = 0, m = this.edges.length;
        ArrayList<long[]> result = new ArrayList<>();

        for (int i = 0; i < m; i++) {
            if (this.edges[(int)start][2] != this.edges[i][2]) {
                long[] groupResult = {start, length, this.edges[(int)start][2]};
                if(length > 1) result.add(groupResult);
                start = i;
                length = 0;
            }
            length++;
        }

        if(length > 1) {
            long[] groupResult = {start, length, this.edges[(int)start][2]};
            result.add(groupResult);
        }

        this.edgeGroups = result.toArray(new long[result.size()][]);
    }

    private void sortEdgeGroups() {
        Arrays.sort(this.edgeGroups, (a, b) -> Long.compare(b[1], a[1]));
    }


    public void clearUF() {
        Arrays.fill(this.ranks, 0);
        Arrays.fill(this.parents, -1);
        this.selectedGroupEdgeCount = 0;
        this.mst = 0;
    }

    public void process() {
        this.sortEdges();
        this.setEdgeGroups();
        this.sortEdgeGroups();
    }

    public void union(int a, int b) {
        int rootA = this.find(a), rootB = this.find(b);
        int rankA = this.ranks[rootA], rankB = this.ranks[rootB];

        if (rankA > rankB) {
            this.parents[rootB] = rootA;
        } else {
            this.parents[rootA] = rootB;
            if (rankA == rankB) this.ranks[rootB] = ++rankB;
        }
    }

    public int find(int id) {
        int root = this.parents[id] == -1 ? id : this.parents[id];
        if (id != root) {
            root = this.find(root);
            this.parents[id] = root;
        }

        return root;
    }

    public boolean connected(int a, int b) {
        return this.find(a) == this.find(b);
    }


    public void runMSTForSelected(long selectedStart, long selectedLength) {
        for (int i = (int)selectedStart; i < selectedStart + selectedLength; i++) {
            long[] e = this.edges[i];
            if (!this.connected((int)e[0], (int)e[1])) {
                long w = e[2];
                this.union((int)e[0], (int)e[1]);
                this.mst += w;
                this.selectedGroupEdgeCount++;
            }
        }
    }

    public long runMST(long limit) {
        for (int i = 0; i < this.edges.length; i++) {
            long[] e = this.edges[i];
            if (!this.connected((int)e[0], (int)e[1])) {
                long w = e[2];
                this.union((int)e[0], (int)e[1]);
                this.mst += w;
                if (this.mst >= limit) return Long.MAX_VALUE;
            }
        }

        return mst;
    }

    public static void main(String[] args) {
        Main hw = new Main();
        hw.runForCLI();
    }
}
