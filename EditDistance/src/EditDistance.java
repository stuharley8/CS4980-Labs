/*
 * Course: Natural Language Processing
 * Spring 2022
 * HW 1
 * Author: Stuart Harley
 * Created: 3/14/2022
 * Reference: https://stackoverflow.com/questions/49900588/edit-distance-java
 */
public class EditDistance {

    private static int minimumEditDistance(String from, String to) {
        return minimumEditDistance(from, to, 0, 0);
    }

    private static int minimumEditDistance(String from, String to, int i, int j) {
        if (j == to.length()) {
            return from.length() - i;
        }
        if (i == from.length()) {
            return to.length() - j;
        }
        if (from.charAt(i) == to.charAt(j))
            return minimumEditDistance(from, to, i + 1, j + 1);
        int del = minimumEditDistance(from, to, i, j + 1) + 1;  // Deletion is worth 1
        int ins = minimumEditDistance(from, to, i + 1, j) + 1;  // Insertion is worth 1
        int rep = minimumEditDistance(from, to, i + 1, j + 1) + 2; // Replacement is worth 2
        return Math.min(rep, Math.min(del, ins));
    }

    public static void main(String[] args) {
        System.out.println("LEDA -> DEAL: " + minimumEditDistance("leda", "deal"));
        System.out.println("DRIVE -> BRIEF: " + minimumEditDistance("drive", "brief"));
        System.out.println("DRIVE -> DIVERS: " + minimumEditDistance("drive", "divers"));
    }
}
