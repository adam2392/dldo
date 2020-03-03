/****************************************************************************/
/*                                                                          */
/*  467/667: Generate Graphs for Hello DL World (HW #2)                     */
/*  Date:  February 25, 2020                                                */
/*                                                                          */
/*  If you would like to change the form of the output, modify the          */
/*  printf statements on the lines marked as OUTPUT.                        */
/*                                                                          */
/****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#define NMAX 8
#define EMAX ((NMAX * (NMAX-1)) / 2)

int main (int ac, char **av);
void stable (int ncount, int ecount, int vec[EMAX], int *alpha);

int main (int ac, char **av)
{
    int rval = 0, ncount = 0, ecount = 0, vcount = 0, i, j;
    int vec[EMAX], alpha, total[NMAX+1]; 

    if (ac < 2) {
        printf ("Usage: %s number_of_nodes\n", av[0]);
        goto CLEANUP;
    }

    ncount = atoi (av[1]);
    if (ncount < 4 || ncount > NMAX) {
        printf ("Must have between 4 and %d nodes\n", NMAX); fflush (stdout);
        goto CLEANUP;
    }
    printf ("All labeled graphs on %d nodes\n", ncount); fflush (stdout);

    for (i = 0; i <= ncount; i++) total[i] = 0;

    ecount = (ncount * (ncount-1)) / 2;
    vcount = 1 << ecount;
    printf ("Number of graphs: %d\n", vcount); fflush (stdout);

    for (i = 0; i < vcount; i++) {
        for (j = 0; j < ecount; j++) {
            vec[j] = (i & (1 << j) ? 1 : 0);
            printf ("%d ", vec[j]);  /* OUTPUT */
        }
        stable (ncount, ecount, vec, &alpha);
        printf ("%d\n", alpha);      /* OUTPUT */
        total[alpha]++;
    }

    printf ("\n");
    for (i = 1; i <= ncount; i++) {
        if (total[i] > 1) {
             printf ("alpha %d: %d graphs\n", i, total[i]);
        } else {
             printf ("alpha %d: %d graph\n", i, total[i]);
        }
        fflush (stdout);
    }

CLEANUP:
    return rval;
}

void stable (int ncount, int ecount, int edge[EMAX], int *alpha) 
{
    int i, j, k, n, yesno, card, scount, xbest = 0, s[NMAX];

    scount = 1 << ncount;

    /* try each subset of nodes */
    for (n = 0; n < scount; n++) {
        card = 0;
        for (i = 0; i < ncount; i++) {
            s[i] = (n & (1 << i) ? 1 : 0);
            card += s[i];
        }

        /* the set of nodes is represted by the 0-1 vector s           */
        /* if the set has more than xbest nodes, check if it is stable */

        if (card > xbest) {  
            /* run through all edges and check if both ends are in s   */
            yesno = 1;
            for (i = 0, k = 0; i < ncount && yesno == 1; i++) {
                for (j = i+1; j < ncount; j++) {
                    if (edge[k++] && s[i] && s[j]) {
                        yesno = 0; break;
                    }
                }
            }
            if (yesno == 1) xbest = card;
        }
    }

    *alpha = xbest;
    return; 
}

