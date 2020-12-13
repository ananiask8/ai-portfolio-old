package pal;

import java.io.*;
import java.math.BigInteger;
import java.util.*;

public class Main {
    private static String BASE_PATH = "/Users/ananias/Documents/CTU/Advanced Algorithms/Labs/06/dataset/";
    private long rangeLow_A;
    private long rangeHigh_B;
    private long rangeLow_C;
    private long rangeHigh_D;
    private int minCompositeRootsOfDegreeF_E;
    private int compositenessDegree_F;
    private ArrayList<Integer> ps;
    private ArrayList<HashMap<Long, Integer>> primeFactorsCD = new ArrayList<>();
    private ArrayList<HashSet<Long>> allFactorsCD = new ArrayList<>();
    private boolean sieveB = false;

    public void readInputFromDataset(String path) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(BASE_PATH + path));
            String line = reader.readLine();
            StringTokenizer st = new StringTokenizer(line, " ");
            rangeLow_A = Long.parseLong(st.nextToken());
            rangeHigh_B = Long.parseLong(st.nextToken());
            rangeLow_C = Long.parseLong(st.nextToken());
            rangeHigh_D = Long.parseLong(st.nextToken());
            minCompositeRootsOfDegreeF_E = Integer.parseInt(st.nextToken());
            compositenessDegree_F = Integer.parseInt(st.nextToken());
            for (int i = 0; i <= rangeHigh_D - rangeLow_C; i++) {
                primeFactorsCD.add(new HashMap<>());
                allFactorsCD.add(new HashSet<>());
            }

            System.out.println("\r\n{ [A: " + rangeLow_A + ", B: " + rangeHigh_B + "], [C: " + rangeLow_C + ", D: " +
                    rangeHigh_D + "], E: " + minCompositeRootsOfDegreeF_E + ", F: " + compositenessDegree_F + "  }");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String readOutputFromDataset(String path) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(BASE_PATH + path));
            return reader.readLine();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return "";
    }

    public void readInputFromCLI() {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            String line = reader.readLine();
            StringTokenizer st = new StringTokenizer(line, " ");
            rangeLow_A = Long.parseLong(st.nextToken());
            rangeHigh_B = Long.parseLong(st.nextToken());
            rangeLow_C = Long.parseLong(st.nextToken());
            rangeHigh_D = Long.parseLong(st.nextToken());
            minCompositeRootsOfDegreeF_E = Integer.parseInt(st.nextToken());
            compositenessDegree_F = Integer.parseInt(st.nextToken());
            for (int i = 0; i <= rangeHigh_D - rangeLow_C; i++) {
                primeFactorsCD.add(new HashMap<>());
                allFactorsCD.add(new HashSet<>());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public long gcd(long x, long y) {
        if(y == 0) return x;
        return gcd(y, x % y);
    }

    // Pollard-Rho's
    // n cannot be a prime
    private long pollardRho(long n) {
        Random rand = new Random();
        if (n == 1) return 1;
        if (n % 2 == 0) return 2;
        long x = rand.longs(1, 2, n - 2).findFirst().getAsLong();
        long c = rand.longs(1, 1, n - 1).findFirst().getAsLong();
        long y = x, d = 1;
        while (d == 1) {
            x = (modPow(x, 2, n) + c + n) % n;
            y = (modPow(y, 2, n) + c + n) % n;
            y = (modPow(y, 2, n) + c + n) % n;

            d = gcd(Math.abs(x - y), n);
            if (d == n) return pollardRho(n);
        }
        return d;
    }

    public void factorizeRho(long n, ArrayList<Long> factors) {
        if (n == 1) return;
        if (isPrime(n, 20)) { factors.add(n); return; }
        long factor = pollardRho(n);
        factorizeRho(factor, factors);
        factorizeRho(n / factor, factors);
    }

    public ArrayList<Long> factorize(long n, ArrayList<Integer> ps) {
        int sqrt_n = (int) Math.sqrt(n);
        ArrayList<Long> factors = new ArrayList<>();
        for (Integer prime: ps) {
            if (prime > sqrt_n) break;
            if (n % prime == 0) factors.add((long) prime);
            while (n % prime == 0) n /= prime;
        }
        factorizeRho(n, factors);
        return factors;
    }

    public void factorizeExp(long n, HashMap<Long, Long> factors) {
        while (true) {
            if (isPrime(n, 128)) {
                if (n != 1) {
                    if (!factors.containsKey(n)) factors.put(n, 0l);
                    factors.put(n, factors.get(n) + 1);
                }
                break;
            }
            long factor = pollardRho(n);
            while (!isPrime(factor, 128)) factor = pollardRho(n);

            n /= factor;
            if (!factors.containsKey(factor)) factors.put(factor, 0l);
            factors.put(factor, factors.get(factor) + 1);
        }
    }

    // Sieve of Eratosthenes
    public ArrayList<Integer> primes(int n) {
        ArrayList<Integer> ps = new ArrayList<>();
        boolean[] sieve = new boolean[n + 1];
        Arrays.fill(sieve, true);

        for (long p = 2; p <= n; p++) {
            if (sieve[(int) p]) ps.add((int) p);
            for (long i = p*p; i <= n; i += p) sieve[(int) i] = false;
        }

        return ps;
    }

    // Segmented Sieve
    // Uses Math.floorMod((low + p + 1) / -2, p) for initially allocating shift from initial index
    // so that can mark all multiples of primes
    public ArrayList<Long> primeRange(long low, long high, int delta, ArrayList<Integer> ps) {
        ArrayList<Long> allPs = new ArrayList<>();
        boolean[] sieve = new boolean[delta];
        int[] qs = new int[ps.size()];
        int s = 0;
        for (int i = 1; i < ps.size(); i++) {
            Integer p = ps.get(i);
            qs[s] = (int) Math.floorMod((low + p + 1) / -2, p);
            s++;
        }
        while (low <= high) {
            Arrays.fill(sieve, true);
            for (int i = qs[0]; i < delta; i += ps.get(1)) sieve[i] = false;
            for (s = 2; s < ps.size(); s++) {
                int p = ps.get(s), q = qs[s - 1];
                for (int i = q; i < delta; i += p) sieve[i] = false;
                qs[s - 1] = Math.floorMod(qs[s - 2] - delta, p);
            }
            long t = low % 2 == 0 ? low + 1 : low;
            for (int i = 0; i < delta && t <= high; i++) {
                if (sieve[i]) allPs.add(t);
                t += 2;
            }
            low += (2 * delta);
        }

        return allPs;
    }

    private long modPow(long base, long exponent, long mod) {
        int shift = 63;
        long result = base;

        while (((exponent >> shift--) & 1) == 0);

        while (shift >= 0) {
            result = (result * result) % mod;
            if (((exponent >> shift--) & 1) == 1) result = (result * base) % mod;
        }

        return result;
    }

    private long pow(long base, long exponent) {
        if (exponent == 0) return 1;
        int shift = 63;
        long result = base;

        while (((exponent >> shift--) & 1) == 0);

        while (shift >= 0) {
            result = result * result;
            if (((exponent >> shift--) & 1) == 1)
                result = result * base;
        }

        return result;
    }

    // Rabbin-Miller Test
    public boolean isPrime(long n, int k) {
        if (n <= 2) return true;
        long r = 0, d = (n - 1) / (2 << r);
        while (d % 2 == 0) d = (n - 1) / pow(2, ++r);

        startOfLoop:
        for (int i = 0; i < k; i++) {
            Random rand = new Random();
            BigInteger a = BigInteger.valueOf(rand.longs(1, 2, n - 2 > 2 ? n - 2 : n).findFirst().getAsLong());
            BigInteger x = a.modPow(BigInteger.valueOf(d), BigInteger.valueOf(n));
            if (x.equals(BigInteger.ONE) || x.equals(BigInteger.valueOf(n-1))) continue startOfLoop;

            for (int j = 0; j < r - 1; j++) {
                x = x.modPow(BigInteger.valueOf(2), BigInteger.valueOf(n));

                if (x.equals(BigInteger.ONE)) return false;
                if (x.equals(BigInteger.valueOf(n - 1))) continue startOfLoop;
            }
            return false;
        }
        return true;
    }


    // This function finds f-composites in range [low, high] by using a sieve up to sqrt(high)
    // It is conjectured that any remainder, after factorizing n with all primes up to sqrt(high) will itself be a prime
    public ArrayList<Long> primitiveRootCandidates(long low, long high, int degreeF) {
        HashMap<Long, HashMap<Long, Long>> range = new HashMap<>();
        for (long n = low; n <= high; n++) range.put(n, new HashMap<>());
        for (long n = low % 2 == 0? low : low + 1; n <= high; n += 2) {
            long x = (long) Math.floor(Math.log(n) / Math.log(2));
            long exp = (int) Math.round(Math.log(gcd(pow(2, x), n)) / Math.log(2));
            range.get(n).put(2l, exp);
        }
        for (int i = 1; i < ps.size(); i++) {
            Long prime = (long) ps.get(i);
            long start = low + Math.floorMod(-low, prime);
                for (long n = start; n <= high; n += prime) {
                    long x = (long) Math.floor(Math.log(n) / Math.log(prime));
                    long exp = Math.round(Math.log(gcd(pow(prime, x), n)) / Math.log(prime));
                    range.get(n).put(prime, exp);
                }
        }
        ArrayList<Long> compositeRoots = new ArrayList<>();
        for (Long n: range.keySet()) {
            Long currentComposite = range.get(n).keySet().stream().reduce(1l, (acc, prime) -> acc * pow( (long) prime, (long) range.get(n).get(prime)));
            Long remainderComposite = n / currentComposite;
            if (currentComposite < n) range.get(n).put(remainderComposite, 1l);
            if (range.get(n).values().stream().reduce(1l, (acc, x) -> acc * (x + 1l)) >= degreeF) compositeRoots.add(n);
        }

        return compositeRoots;
    }

    // Let PFM be the set of all prime factors of M−1. Then R is the primitive root of M if and only if
    // R(M−1)/p is not congruent to 1 mod M, for all p ∈ PFM.
    public boolean primitiveRootsIn(long n, Set<Long> factors, ArrayList<Long> in, int limit) {
        int count = 0;
        int i = 0;
        nextComposite:
        for (long composite: in) {
            if (limit - count > in.size() - i++) return false;
            for (long factor: factors) {
                if (BigInteger.valueOf(composite).modPow(BigInteger.valueOf((n - 1) / factor), BigInteger.valueOf(n)).equals(BigInteger.ONE)) {
                    continue nextComposite;
                }
            }
            if(++count >= limit) return true;
        }

        return count >= limit;
    }


    // Generates primes for range [low, high] using Rabbin Miller's primality test
    public ArrayList<Long> primeRange(long low, long high) {
        ArrayList<Long> result = new ArrayList<>();
        for (long n = low; n <= high; n++) {
            if (isPrime(n, 128)) result.add(n);
        }
        return result;
    }

    // This function finds factors for prime - 1 of a parameter array primes_AB
    // It uses a sieve up to sqrt(value near B), and then factorizes the rest by pollardRho, since it can't make guarantees unless under some conditions
    // It is conjectured that any remainder, after factorizing n with all primes up to sqrt(high) will itself be a prime
    public void getFactorsForPrimes(ArrayList<Long> primes_AB, HashMap<Long, Set<Long>> factors) {
        HashMap<Long, HashMap<Long, Long>> currentFactors = new HashMap<>();
        for (long prime: primes_AB) currentFactors.put(prime - 1, new HashMap<>());
        for (long prime: ps) {
            for (long p: primes_AB) {
                long q = p - 1;
                if (q % prime != 0) continue;
                long x = (long) Math.floor(Math.log(q) / Math.log(prime));
                long exp = Math.round(Math.log(gcd(pow(prime, x), q)) / Math.log(prime));
                currentFactors.get(q).put(prime, exp);
            }
        }

        for (Long n: currentFactors.keySet()) {
            Long currentComposite = currentFactors.get(n).keySet().stream().reduce(1l, (acc, prime) -> acc * pow(prime, currentFactors.get(n).get(prime)));
            Long remainderComposite = n / currentComposite;
            if (currentComposite < n && remainderComposite < ps.get(ps.size() - 1)) currentFactors.get(n).put(remainderComposite, 1l);
            else if (remainderComposite > ps.get(ps.size() - 1)) factorizeExp(remainderComposite, currentFactors.get(n));
            factors.put(n, currentFactors.get(n).keySet());
        }

    }

    public long[] solve(boolean withTimes) {
        sieveB |= Math.sqrt(rangeHigh_D) < 1e6;
        long t1 = System.nanoTime();

        ps = primes((int) Math.sqrt(sieveB ? rangeHigh_B : rangeHigh_D));

        long t2 = System.nanoTime();
        if (withTimes) System.out.println("Primes up to sqrt(" + (sieveB ? "B" : "D") + "): " + (t2 - t1)/ 1000000000.0);
        t1 = t2;

        ArrayList<Long> primes_AB = sieveB ? primeRange(rangeLow_A, rangeHigh_B, (int) (rangeHigh_B - rangeLow_A), ps) : primeRange(rangeLow_A, rangeHigh_B);

        t2 = System.nanoTime();
        if (withTimes) System.out.println("Primes in range [A, B]: " + (t2 - t1)/ 1000000000.0);
        t1 = t2;

        ArrayList<Long> composite_CD = primitiveRootCandidates(rangeLow_C, rangeHigh_D, compositenessDegree_F);
//        System.out.println(composite_CD);

        t2 = System.nanoTime();
        if (withTimes) System.out.println("F-Composite numbers in range [C, D]: " + (t2 - t1)/ 1000000000.0);
        t1 = t2;

        ArrayList<Long> result = new ArrayList<>();
        HashMap<Long, Set<Long>> factors = new HashMap<>();
        getFactorsForPrimes(primes_AB, factors);
        for (long p: primes_AB) {
            if (primitiveRootsIn(p, factors.get(p - 1), composite_CD, minCompositeRootsOfDegreeF_E)) result.add(p);
        }

        t2 = System.nanoTime();
        if (withTimes) System.out.println("Primes with at least E primitive roots in range [C, D]: " + (t2 - t1)/ 1000000000.0);
        t1 = t2;
        BigInteger sum = BigInteger.valueOf(0l);
        BigInteger prod = BigInteger.valueOf(1l);
        for (long val: result) {
            sum = sum.add(BigInteger.valueOf(val));
            prod = prod.multiply(BigInteger.valueOf(val));
        }

//        System.out.println(result);
        if (withTimes) System.out.println("Calculating result: " + (t2 - t1)/ 1000000000.0);
        return new long[] {result.size(), prod.mod(sum).longValue()};
    }

    public long[] runFor(String env, String path) {
        if (!env.equals("dataset")) return new long[] {};
        readInputFromDataset(path);

        return solve(true);
    }

    public long[] runFor(String env) {
        if (!env.equals("cli")) return new long[] {};
        readInputFromCLI();

        return solve(false);
    }

    public static void test() {
        for (int i = 1; i <= 10; i++) {
            long startTime = System.nanoTime();

            Main s = new Main();
            String filename = "pub" + (i >= 10 ? i : "0" + i);
            long[] result = s.runFor("dataset", filename + ".in");
            String[] expected = s.readOutputFromDataset(filename + ".out").split(" ");

            long stopTime = System.nanoTime();
            System.out.println("total time: " + (stopTime - startTime)/ 1000000000.0);
            String checkMark = result[0] == Long.parseLong(expected[0]) && result[1] == Long.parseLong(expected[1])  ? "✔" : "✘";
            System.out.println(checkMark + " - { expected: [ " + expected[0] + " " + expected[1] + " ], result: [ " + result[0] + " " + result[1] + " ] };\r\n");
        }
    }

    public static void conquer() {
        Main s = new Main();
        long[] result = s.runFor("cli");
        System.out.println(result[0] + " " + result[1]);
    }

    public static void main(String[] args) {
        // Main.test();
       Main.conquer();
    }
}