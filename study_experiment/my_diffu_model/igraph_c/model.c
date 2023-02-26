#include <igraph.h>
#include "func.h"
#include <stdio.h>
#include <sys/timeb.h>
#include <math.h>

void init(model *M, const igraph_t G, double au_T_rate)
{

    /*
    Group value:
        inactive: 0
        R-active: 1
        T-active: 2

    */

    igraph_copy(&(M->G), &G);

    // init group value
    igraph_vector_t group;
    igraph_vector_init(&group, igraph_vcount(&G));
    igraph_cattribute_VAN_setv(&(M->G), "group", &group);
    igraph_vector_destroy(&group);

    // init influenced threshold & correction threshold
    igraph_vit_t vit;
    igraph_vit_create(&(M->G), igraph_vss_all(), &vit);

    struct timeval start;
    gettimeofday(&start, NULL);                           // need microsecond
    igraph_rng_seed(igraph_rng_default(), start.tv_usec); // init random number generator seed
    while (!IGRAPH_VIT_END(vit))
    {
        igraph_integer_t vid = IGRAPH_VIT_GET(vit);
        igraph_cattribute_VAN_set(&(M->G), "ithr", vid, igraph_rng_get_unif01(igraph_rng_default()));
        igraph_cattribute_VAN_set(&(M->G), "cthr", vid, igraph_rng_get_unif01(igraph_rng_default()));
        IGRAPH_VIT_NEXT(vit);
    }
    igraph_vit_destroy(&vit);

    // init model's node_degree_vector
    igraph_vector_int_init(&(M->node_degree_vector), 0);
    igraph_degree(&(M->G), &(M->node_degree_vector), igraph_vss_all(), IGRAPH_OUT, IGRAPH_NO_LOOPS);

    // select authoritative T nodes & update group information
    init_auT_nodes(M, au_T_rate);

    // update correction thresholds for nodes which under the influence by au_T nodes
    new_correction_threshold(M, 1, 3);
}

void init_auT_nodes(model *M, double au_T_rate)
{
    /* get median degree */
    igraph_vector_int_t all_degree;
    igraph_vector_int_init_copy(&all_degree, &(M->node_degree_vector));
    igraph_vector_int_sort(&all_degree);
    int median_deg = 0;
    if (igraph_vector_int_size(&all_degree) % 2 == 0)
    {
        median_deg = (VECTOR(all_degree)[igraph_vector_int_size(&all_degree) / 2] + VECTOR(all_degree)[igraph_vector_int_size(&all_degree) / 2 - 1]) / 2;
    }
    else
    {
        median_deg = VECTOR(all_degree)[igraph_vector_int_size(&all_degree) / 2];
    }
    igraph_vector_int_destroy(&all_degree);

    /* random select node which degree greater median degree as authoritative T node */
    int num_of_select = (int)(au_T_rate * igraph_vcount(&(M->G)) / 2);
    if (num_of_select < 1)
    {
        num_of_select = 1;
    }
    
    igraph_vector_int_init(&(M->authoritative_T_vector), num_of_select);
    igraph_vector_int_fill(&(M->authoritative_T_vector), -1);
    struct timeval start;
    gettimeofday(&start, NULL);                           // need microsecond
    igraph_rng_seed(igraph_rng_default(), start.tv_usec); // init random number generator seed
    for (int i = 0; i < num_of_select;)
    {
        int node = igraph_rng_get_integer(igraph_rng_default(), 0, igraph_vcount(&(M->G)) - 1);
        if (igraph_vector_int_get(&(M->node_degree_vector), node) > median_deg)
        {
            if (!igraph_vector_int_contains(&(M->authoritative_T_vector), node))
            {
                igraph_vector_int_set(&(M->authoritative_T_vector), i, node);
                igraph_cattribute_VAN_set(&(M->G), "group", node, 2);
                i++;
            }
            else
            {
                continue;
            }
        }
    }
}

void new_correction_threshold(model *M, double beta, int order)
{
    igraph_vs_t vs;
    igraph_vector_int_list_t nbr;
    igraph_vector_int_list_init(&nbr, 0);

    for (int i = 0; i < igraph_vector_int_size(&(M->authoritative_T_vector)); i++)
    {
        igraph_vs_1(&vs, VECTOR(M->authoritative_T_vector)[i]);
        for (int j = 1; j <= order; j++)
        {
            igraph_neighborhood(&(M->G), &nbr, vs, j, IGRAPH_OUT, j - 1);
            igraph_vector_int_t *temp = igraph_vector_int_list_get_ptr(&nbr, 0);
            for (int k = 0; k < igraph_vector_int_size(temp); k++)
            {
                double p = igraph_cattribute_VAN(&(M->G), "cthr", igraph_vector_int_get(temp, k));
                p = p - p / (1 + exp(beta * order));
                igraph_cattribute_VAN_set(&(M->G), "cthr", igraph_vector_int_get(temp, k), p);
            }
            igraph_vector_int_destroy(temp);
        }
    }
    igraph_vector_int_list_destroy(&nbr);
    igraph_vs_destroy(&vs);
}

void refresh_i_c_thr(model *M, const igraph_t original_G, double au_T_rate)
{
    igraph_vector_int_destroy(&(M->authoritative_T_vector));
    igraph_vector_int_destroy(&(M->node_degree_vector));
    igraph_destroy(&(M->G));
    init(M, original_G, au_T_rate);
}

int check_i_thr(const igraph_t *G, igraph_integer_t node)
{
    /*
    Returns
        int
            0 : nothing change
            1 : activated by rumor
            2 : activated by truth
    */
    double T_num = 0;
    double R_num = 0;
    igraph_integer_t node_deg = 0;
    igraph_degree_1(G, &node_deg, node, IGRAPH_OUT, IGRAPH_NO_LOOPS);

    if (node_deg <= 0)
    {
        return 0;
    }

    igraph_vector_int_t nbr;
    igraph_vector_int_init(&nbr, 0);
    igraph_neighbors(G, &nbr, node, IGRAPH_OUT);
    for (int i = 0; i < igraph_vector_int_size(&nbr); i++)
    {
        if (VAN(G, "group", VECTOR(nbr)[i]) == 1)
        {
            R_num++;
        }
        else if (VAN(G, "group", VECTOR(nbr)[i]) == 2)
        {
            T_num++;
        }
    }
    igraph_vector_int_destroy(&nbr);

    // Priority to become a rumor node
    if ((R_num / node_deg) >= VAN(G, "ithr", node))
    {
        return 1;
    }
    else if ((R_num / node_deg) >= VAN(G, "ithr", node))
    {
        return 2;
    }

    return 0;
}

bool check_c_thr(const igraph_t *G, igraph_integer_t node)
{
    double activated_num = 0;
    double T_active_num = 0;

    igraph_vector_int_t nbr;
    igraph_vector_int_init(&nbr, 0);
    igraph_neighbors(G, &nbr, node, IGRAPH_OUT);
    for (int i = 0; i < igraph_vector_int_size(&nbr); i++)
    {
        if (VAN(G, "group", VECTOR(nbr)[i]) != 0)
        {
            activated_num++;
        }
        if (VAN(G, "group", VECTOR(nbr)[i]) == 2)
        {
            T_active_num++;
        }
    }
    igraph_vector_int_destroy(&nbr);

    if (activated_num == 0)
    {
        return false;
    }
    else if ((T_active_num / activated_num) >= VAN(G, "cthr", node))
    {
        return true;
    }
    return false;
}

void generate_R_nodes(const model M, double R_rate, igraph_vector_int_t *R_nodes)
{
    /* The generated R nodes will be stored in R_nodes */
    int available_num = igraph_vcount(&(M.G)) - igraph_vector_int_size(&(M.authoritative_T_vector));
    int select_num = (int)(available_num * R_rate);
    if (select_num < 1)
    {
        select_num = 1;
    }
    // igraph_vector_int_t R_nodes;
    igraph_vector_int_init(R_nodes, select_num);

    struct timeval start;
    gettimeofday(&start, NULL);                           // need microsecond
    igraph_rng_seed(igraph_rng_default(), start.tv_usec); // init random number generator seed
    for (int i = 0; i < select_num;)
    {
        int node = igraph_rng_get_integer(igraph_rng_default(), 0, igraph_vcount(&(M.G)) - 1);

        if (!igraph_vector_int_contains(R_nodes, node))
        {
            if (!igraph_vector_int_contains(&(M.authoritative_T_vector), node))
            {
                igraph_vector_int_set(R_nodes, i, node);
                i++;
            }
        }
        else
        {
            continue;
        }
    }
    // return R_nodes;
}

void before_detected_diffusion(const model M, igraph_t *G, const igraph_vector_int_t *R_nodes, igraph_vector_int_t *T_recv, igraph_vector_int_t *R_recv, int *spread_time)
{
    // igraph_t G;
    igraph_copy(G, &(M.G));
    // igraph_vector_int_init(T_recv, igraph_vcount(G));
    // igraph_vector_int_fill(T_recv, -1);
    // igraph_vector_int_init(R_recv, igraph_vcount(G));
    // igraph_vector_int_fill(R_recv, -1);
    igraph_vector_int_init(T_recv,0);
    igraph_vector_int_init(R_recv,0);

    igraph_dqueue_int_t queue;
    igraph_dqueue_int_init(&queue, igraph_vcount(G));

    // init T_recv
    for (int i = 0; i < igraph_vector_int_size(&M.authoritative_T_vector); i++)
    {
        // igraph_vector_int_set(T_recv, i, VECTOR(M.authoritative_T_vector)[i]);
        igraph_vector_int_push_back(T_recv,VECTOR(M.authoritative_T_vector)[i]);
    }

    // init R_recv & search range
    igraph_vector_t v;
    igraph_vector_init(&v, igraph_vcount(G));
    SETVANV(G, "actTime", &v);
    igraph_vector_destroy(&v);
    int R_nodes_num = igraph_vector_int_size(R_nodes);

    for (int i = 0; i < igraph_vector_int_size(R_nodes); i++)
    {
        // igraph_vector_int_set(R_recv, i, VECTOR(*R_nodes)[i]);
        igraph_integer_t node = igraph_vector_int_get(R_nodes,i);
        igraph_vector_int_push_back(R_recv,node);
        igraph_cattribute_VAN_set(G, "group", node, 1);

        igraph_vector_int_t nbr;
        igraph_vector_int_init(&nbr, 0);
        igraph_neighbors(G, &nbr, node, IGRAPH_OUT);
        for (int j = 0; j < igraph_vector_int_size(&nbr); j++)
        {
            if (VAN(G, "group", VECTOR(nbr)[j]) != 1) // including T-active nodes
            {
                igraph_dqueue_int_push(&queue, VECTOR(nbr)[j]);
            }
        }
        igraph_vector_int_destroy(&nbr);
    }

    bool nothing_change = false;
    bool is_pause = false;

    while (!nothing_change && !is_pause)
    {
        nothing_change = true;
        int circ_times = igraph_dqueue_int_size(&queue);
        *spread_time = *spread_time + 1;

        for (int i = 0; i < circ_times; i++)
        {
            int node = igraph_dqueue_int_pop(&queue);
            if (VAN(G, "group", node) == 1)
            {
                continue;
            }
            if (VAN(G, "group", node) == 2)
            {
                is_pause = true;
                continue;
            }
            if (check_i_thr(G, node) == 1) // activated by rumor
            {
                nothing_change = false;
                SETVAN(G, "group", node, 1);
                SETVAN(G, "actTime", node, *spread_time);
                // igraph_vector_int_set(R_recv, R_nodes_num, node);
                igraph_vector_int_push_back(R_recv,node);
                R_nodes_num++;

                if (!is_pause)
                {
                    igraph_vector_int_t nbr;
                    igraph_vector_int_init(&nbr, 0);
                    igraph_neighbors(G, &nbr, node, IGRAPH_OUT);
                    for (int j = 0; j < igraph_vector_int_size(&nbr); j++)
                    {
                        if (VAN(G, "group", VECTOR(nbr)[j]) != 1) // including T-active nodes
                        {
                            igraph_dqueue_int_push(&queue, VECTOR(nbr)[j]);
                        }
                    }
                    igraph_vector_int_destroy(&nbr);
                }
            }
        }
        // the spreading stopped before detection
        if (nothing_change && !is_pause)
        {
            struct timeval start;
            gettimeofday(&start, NULL);
            igraph_rng_seed(igraph_rng_default(), start.tv_usec);
            
            // random select a node as R-node
            int rn_node = 0;
            while (true)
            {
                rn_node = igraph_rng_get_integer(igraph_rng_default(), 0, igraph_vcount(G) - 1);
                if ((!igraph_vector_int_contains(R_recv, rn_node)) && (!igraph_vector_int_contains(&M.authoritative_T_vector, rn_node)))
                {
                    break;
                }
            }
            SETVAN(G, "group", rn_node, 1);
            SETVAN(G, "actTime", rn_node, *spread_time);
            // igraph_vector_int_set(R_recv, R_nodes_num, rn_node);
            igraph_vector_int_push_back(R_recv,rn_node);
            R_nodes_num++;

            //update search range
            igraph_vector_int_t nbr;
            igraph_vector_int_init(&nbr, 0);
            igraph_neighbors(G, &nbr, rn_node, IGRAPH_OUT);
            for (int j = 0; j < igraph_vector_int_size(&nbr); j++)
            {
                if (VAN(G, "group", VECTOR(nbr)[j]) != 1) // including T-active nodes
                {
                    igraph_dqueue_int_push(&queue, VECTOR(nbr)[j]);
                }
            }
            igraph_vector_int_destroy(&nbr);

            nothing_change = false;
        }
    }
    igraph_dqueue_int_destroy(&queue);
    // return G;
}

void after_detected_diffusion(const igraph_t *res1, const igraph_vector_int_t *T_nodes, const igraph_vector_int_t *R_nodes, igraph_vector_int_t *T_recv, igraph_vector_int_t *R_recv, int *spread_time)
{
    igraph_t G;
    igraph_copy(&G, res1);

    igraph_dqueue_int_t spr_search_range;
    igraph_dqueue_int_t cor_search_range;
    igraph_dqueue_int_init(&spr_search_range, igraph_vcount(&G));
    igraph_dqueue_int_init(&cor_search_range, igraph_vcount(&G));

    int T_nodes_num = igraph_vector_int_size(T_recv);
    int R_nodes_num = igraph_vector_int_size(R_recv);

    // init T_nodes group attribute
    for (int i = 0; i < igraph_vector_int_size(T_nodes); i++)
    {
        igraph_integer_t node = igraph_vector_int_get(T_nodes,i);
        igraph_cattribute_VAN_set(&G,"group",node,2);
        igraph_vector_int_push_back(T_recv,node);
        T_nodes_num++;
    }

    // init spreading & correction search range
    igraph_vector_int_t nbr;
    igraph_vector_int_init(&nbr, 0);
    for (int i = 0; i < igraph_vector_int_size(R_recv); i++)
    {
        igraph_neighbors(&G, &nbr, igraph_vector_int_get(R_recv, i), IGRAPH_OUT);
        for (int j = 0; j < igraph_vector_int_size(&nbr); j++)
        {

            if (VAN(&G, "group", VECTOR(nbr)[j]) == 0) // the nbr is inactive
            {
                igraph_dqueue_int_push(&spr_search_range, VECTOR(nbr)[j]);
            }
        }
    }
    igraph_vector_int_destroy(&nbr);
    
    igraph_vector_int_init(&nbr, 0);
    for (int i = 0; i < igraph_vector_int_size(T_recv); i++)
    {
        // if (igraph_vector_int_get(T_recv, i) == -1)
        // {
        //     break;
        // }
        igraph_neighbors(&G, &nbr, igraph_vector_int_get(T_recv, i), IGRAPH_OUT);
        for (int j = 0; j < igraph_vector_int_size(&nbr); j++)
        {
            if (VAN(&G, "group", VECTOR(nbr)[j]) == 0) // the nbr is inactive
            {
                igraph_dqueue_int_push(&spr_search_range, VECTOR(nbr)[j]);
            }
            else if (VAN(&G, "group", VECTOR(nbr)[j]) == 1) // the nbr is R-active
            {
                igraph_dqueue_int_push(&cor_search_range, VECTOR(nbr)[j]);
            }
        }
    }
    igraph_vector_int_destroy(&nbr);

    // start diffusion
    bool nothing_change = false;
    while (!nothing_change)
    {
        nothing_change = true;
        int spr_circ_times = igraph_dqueue_int_size(&spr_search_range);
        int cor_circ_times = igraph_dqueue_int_size(&cor_search_range);
        *spread_time = *spread_time + 1;

        // the phase of T&R spreading
        for (int i = 0; i < spr_circ_times; i++)
        {
            int node = igraph_dqueue_int_pop(&spr_search_range);
            if (VAN(&G, "group", node) != 0)
            {
                continue;
            }
            int check_code = check_i_thr(&G, node);
            if (check_code == 1) // activated by rumor
            {
                nothing_change = false;
                SETVAN(&G, "group", node, 1);
                SETVAN(&G, "actTime", node, *spread_time);
                // igraph_vector_int_set(R_recv, R_nodes_num, node);
                igraph_vector_int_push_back(R_recv,node);
                R_nodes_num++;

                // update spread search range
                igraph_vector_int_t nbr;
                igraph_vector_int_init(&nbr, 0);
                igraph_neighbors(&G, &nbr, node, IGRAPH_OUT);
                for (int j = 0; j < igraph_vector_int_size(&nbr); j++)
                {
                    if (VAN(&G, "group", VECTOR(nbr)[j]) == 0)
                    {
                        igraph_dqueue_int_push(&spr_search_range, VECTOR(nbr)[j]);
                    }
                }
                igraph_vector_int_destroy(&nbr);
            }
            else if (check_code == 2) // activated by truth
            {
                nothing_change = false;
                SETVAN(&G, "group", node, 2);
                // igraph_vector_int_set(T_recv, T_nodes_num, node);
                igraph_vector_int_push_back(T_recv,node);
                T_nodes_num++;

                igraph_vector_int_t nbr;
                igraph_vector_int_init(&nbr, 0);
                igraph_neighbors(&G, &nbr, node, IGRAPH_OUT);
                for (int j = 0; j < igraph_vector_int_size(&nbr); j++)
                {
                    if (VAN(&G, "group", VECTOR(nbr)[j]) == 0) // update spread search range
                    {
                        igraph_dqueue_int_push(&spr_search_range, VECTOR(nbr)[j]);
                    }
                    else if (VAN(&G, "group", VECTOR(nbr)[j]) == 1) // update correction search range
                    {
                        igraph_dqueue_int_push(&cor_search_range, VECTOR(nbr)[j]);
                    }
                }
                igraph_vector_int_destroy(&nbr);
            }
            else //nothing happened
            {
                igraph_dqueue_int_push(&spr_search_range,node);
            }
        }

        // the phase of correction
        for (int i = 0; i < cor_circ_times; i++)
        {
            int node = igraph_dqueue_int_pop(&cor_search_range);
            if (VAN(&G, "group", node) != 1)
            {
                continue;
            }
            if (igraph_vector_int_contains(R_nodes, node))
            {
                continue;
            }
            if (check_c_thr(&G, node))
            {
                SETVAN(&G, "group", node, 2);
                nothing_change = false;
                // igraph_vector_int_set(T_recv, T_nodes_num, node);
                igraph_vector_int_push_back(T_recv,node);
                T_nodes_num++;

                // remove R node from R_recv
                igraph_integer_t pos;
                if (igraph_vector_int_search(R_recv, 0, node, &pos))
                {
                    igraph_vector_int_remove(R_recv, pos);
                    R_nodes_num--;
                }
    
                igraph_vector_int_t nbr;
                igraph_vector_int_init(&nbr, 0);
                igraph_neighbors(&G, &nbr, node, IGRAPH_OUT);
                for (int j = 0; j < igraph_vector_int_size(&nbr); j++)
                {
                    if (VAN(&G, "group", VECTOR(nbr)[j]) == 0) // update spread search range
                    {
                        igraph_dqueue_int_push(&spr_search_range, VECTOR(nbr)[j]);
                    }
                    else if (VAN(&G, "group", VECTOR(nbr)[j]) == 1) // update correction search range
                    {
                        igraph_dqueue_int_push(&cor_search_range, VECTOR(nbr)[j]);
                    }
                }
                igraph_vector_int_destroy(&nbr);
            }
            else // nothing happened
            {
                igraph_dqueue_int_push(&cor_search_range,node);
            }
        }
    }
    igraph_dqueue_int_destroy(&spr_search_range);
    igraph_dqueue_int_destroy(&cor_search_range);
    igraph_destroy(&G);
}

// void after_detected_diffusion_2(const model M,const igraph_t *res1, const igraph_vector_int_t T_nodes, const igraph_vector_int_t R_nodes, igraph_vector_int_t *T_recv, igraph_vector_int_t *R_recv, int *spread_time)
// int main()
// {
//     model M;
//     igraph_t G;
//     igraph_set_attribute_table(&igraph_cattribute_table);
//     /* read Graph from gml */
//     FILE *ifile;
//     ifile = fopen("../dataset/dolphins.gml", "r");
//     if (ifile == 0)
//     {
//         printf("No such file\n");
//         return 1;
//     }
//     igraph_read_graph_gml(&G, ifile);
//     fclose(ifile);

//     init(&M, G, 0.01);
//     printf("authoritative T node:\n");
//     for (int i = 0; i < igraph_vector_int_size(&M.authoritative_T_vector); i++)
//     {
//         printf("node id: %ld, group: %d \n", VECTOR(M.authoritative_T_vector)[i], (int)VAN(&M.G, "group", VECTOR(M.authoritative_T_vector)[i]));
//     }
//     igraph_vector_t attr;
//     igraph_vector_init(&attr, 0);
//     VANV(&M.G, "group", &attr);
//     printf("ALL nodes:\n");
//     for (int i = 0; i < igraph_vcount(&M.G); i++)
//     {
//         printf("node id: %d, group: %d, ithr:%.3f, cthr:%.3f\n", i, (int)VECTOR(attr)[i],VAN(&M.G,"ithr",i),VAN(&M.G,"cthr",i));
//     }

//     printf("R_nodes:\n");
//     igraph_vector_int_t r_node;
//     igraph_vector_int_init(&r_node, 0);
//     r_node = generate_R_nodes(M, 0.08);
//     for (int i = 0; i < igraph_vector_int_size(&r_node); i++)
//     {
//         printf("node id: %ld\n", VECTOR(r_node)[i]);
//     }
//     printf("\n");

//     igraph_vector_int_t T_recv;
//     igraph_vector_int_t R_recv;
//     int spread_time = 0;
//     igraph_t res1;
//     igraph_t res2;
//     res1 = before_detected_diffusion(M, r_node, &T_recv, &R_recv, &spread_time);
//     printf("R_recv_size:%d, T_recv_size:%d\n",count_vector_size(&R_recv),count_vector_size(&T_recv));
//     printf("first detected time: %d, R receivers:\n",spread_time);
//     for (int i = 0; i < igraph_vector_int_size(&R_recv); i++)
//     {
//         if (VECTOR(R_recv)[i] == -1)
//         {
//             printf("meet -1\n");
//             break;
//         }
//         printf("node id: %ld, group: %d \n", VECTOR(R_recv)[i], (int)VAN(&res1, "group", VECTOR(R_recv)[i]));
//     }

//     igraph_vector_int_t T_nodes;
//     igraph_vector_int_init(&T_nodes, 6);

//     T_nodes = greedy(6,M,res1,T_recv,R_recv,r_node,spread_time);
//     printf("selected T nodes:\n");
//     for (int i = 0; i < igraph_vector_int_size(&T_nodes); i++)
//     {
//         printf("node id: %ld\n", VECTOR(T_nodes)[i]);
//     }
//     printf("\n");

//     res2 = after_detected_diffusion(M,&res1, T_nodes, r_node, &T_recv, &R_recv, &spread_time);
//     printf("------------------- spreading finish ---------------------------\n");
//     printf("finish time: %d, R receivers:\n",spread_time);
//     for (int i = 0; i < igraph_vector_int_size(&R_recv); i++)
//     {
//         if (VECTOR(R_recv)[i] == -1)
//         {
//             printf("meet -1\n");
//             break;
//         }
//         printf("node id: %ld, group: %d \n", VECTOR(R_recv)[i], (int)VAN(&res2, "group", VECTOR(R_recv)[i]));
//     }
//     printf("T receivers:\n");
//     for (int i = 0; i < igraph_vector_int_size(&T_recv); i++)
//     {
//         if (VECTOR(T_recv)[i] == -1)
//         {
//             printf("meet -1\n");
//             break;
//         }
//         printf("node id: %ld, group: %d \n", VECTOR(T_recv)[i], (int)VAN(&res2, "group", VECTOR(T_recv)[i]));
//     }

//     printf("ALL nodes:\n");
//     VANV(&res2, "group", &attr);
//     for (int i = 0; i < igraph_vcount(&res2); i++)
//     {
//         printf("node id: %d, group: %d, ithr:%.3f, cthr:%.3f\n", i, (int)VECTOR(attr)[i],VAN(&res2,"ithr",i),VAN(&res2,"cthr",i));
//     }
//     igraph_vector_destroy(&attr);

//     // igraph_vector_t attr;
//     // igraph_vector_init(&attr, 0);
//     // VANV(&res2, "group", &attr);
//     // printf("ALL nodes:\n");
//     // for (int i = 0; i < igraph_vcount(&res2); i++)
//     // {
//     //     printf("node id: %d, group: %d, ithr:%.3f, cthr:%.3f\n", i, (int)VECTOR(attr)[i],VAN(&res2,"ithr",i),VAN(&res2,"cthr",i));
//     // }
//     // igraph_vector_destroy(&attr);
//     igraph_vector_int_destroy(&T_recv);
//     igraph_vector_int_destroy(&R_recv);
//     igraph_vector_int_destroy(&T_nodes);
//     igraph_vector_int_destroy(&r_node);

//     igraph_vector_int_destroy(&(M.authoritative_T_vector));
//     igraph_vector_int_destroy(&(M.node_degree_vector));
//     igraph_destroy(&(M.G));
//     igraph_destroy(&G);
//     igraph_destroy(&res1);
//     igraph_destroy(&res2);

//     return 0;
// }