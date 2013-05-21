#include <QCoreApplication>
#include <gsl/gsl_linalg.h>
#include <QFile>
#include <QTextStream>
#include <QDebug>

void matrix_mul(gsl_matrix *result, const gsl_matrix *A, const gsl_matrix *B)
{
    double data;
    size_t jR = 0;
    for(size_t iA = 0; iA < A->size1; ++ iA)
    {
        for(size_t jA = 0; jA < A->size2; ++ jA)
        {
            data = 0;
            for(size_t iB = 0; iB < B->size1; ++ iB)
            {
                data += gsl_matrix_get(A, iA, jA) * gsl_matrix_get(B, iB, jA);
            }
            gsl_matrix_set(result, iA, jR ++, data);
        }
        jR = 0;
    }
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    
    const size_t n1 = 13,
            n2 = 9,
            m1 = 9,
            m2 = 9,
            k = 9,
            h = 9;

    double array[n1][n2] =
     {
      {1,0,0,1,0,1,0,1,0},
      {0,0,0,1,0,0,0,1,0},
      {0,0,0,1,0,0,0,1,0},
      {0,0,1,0,1,0,0,0,1},
      {0,0,1,0,1,0,0,0,1},
      {1,0,0,1,0,1,0,1,0},
      {1,0,0,0,0,0,0,1,0},
      {0,0,1,0,1,0,0,0,1},
      {0,1,0,0,0,0,1,0,0},
      {0,0,1,0,0,0,1,0,0},
      {0,1,0,0,0,1,0,0,0},
      {0,1,0,0,0,0,1,0,0},
      {0,0,1,0,1,0,0,0,0}
     };


    gsl_matrix *A = gsl_matrix_alloc(n1, n2);
    gsl_matrix *V = gsl_matrix_alloc(m1, m2);
    gsl_vector *S = gsl_vector_alloc(k);
    gsl_vector *work = gsl_vector_alloc(h);


    for(size_t i = 0; i < n1; ++ i)
    {
        for(size_t j = 0; j < n2; ++ j)
        {
            gsl_matrix_set(A, i, j, array[i][j]);
        }
    }

    gsl_linalg_SV_decomp(A,V, S, work);

    QFile *out = new QFile("../svd_gnu/out");
    out->open(QIODevice::Text | QIODevice::WriteOnly | QIODevice::Truncate);

    if(out->isOpen())
    {
        QTextStream out_stream(out);
        out_stream << "A\n";
        out_stream.setFieldWidth(13);
        for(size_t i = 0; i < n1; ++ i)
        {
            for(size_t j = 0; j < n2; ++ j)
            {
                out_stream << gsl_matrix_get(A, i, j);
            }
            out_stream << '\n';
        }

        out_stream << "\nW\n";
        for(size_t i = 0; i < k; ++ i)
        {
            out_stream << gsl_vector_get(S, i);
        }

        out_stream << "\nV\n";
        for(size_t i = 0; i < m1; ++ i)
        {
            for(size_t j = 0; j < m2; ++ j)
            {
                out_stream << gsl_matrix_get(V, i, j);
            }
            out_stream << '\n';
        }

        size_t nn = 13, mm = 3;

        // A - U, W - S, V - V

        out_stream << "\nPerfom to mul.\n\nA\n";

        gsl_matrix *Am = gsl_matrix_alloc(nn, mm);
        gsl_matrix *Wm = gsl_matrix_alloc(3, 3);
        gsl_matrix *Vm = gsl_matrix_alloc(3, 9);

        for(size_t i = 0; i < nn; ++ i)
        {
            for(size_t j = 0; j < mm; ++ j)
            {
                gsl_matrix_set(Am, i, j, gsl_matrix_get(A, i, j));
                out_stream << gsl_matrix_get(A, i, j);
            }
            out_stream << "\n";
        }

        out_stream << "\nS\n";

        gsl_matrix_set(Wm, 0, 0, gsl_vector_get(S, 0));
        gsl_matrix_set(Wm, 1, 1, gsl_vector_get(S, 1));
        gsl_matrix_set(Wm, 2, 2, gsl_vector_get(S, 2));

        gsl_matrix_set(Wm, 0, 1, 0);
        gsl_matrix_set(Wm, 1, 0, 0);
        gsl_matrix_set(Wm, 2, 0, 0);
        gsl_matrix_set(Wm, 0, 2, 0);
        gsl_matrix_set(Wm, 1, 2, 0);
        gsl_matrix_set(Wm, 2, 1, 0);

        for(size_t i = 0; i < 3; ++ i)
        {
            for(size_t j = 0; j < 3; ++ j)
            {
                out_stream << gsl_matrix_get(Wm, i, j);
            }
            out_stream << "\n";
        }

        out_stream << "\nV\n";

        for(size_t i = 0; i < 3; ++ i)
        {
            for(size_t j = 0; j < 9; ++ j)
            {
                gsl_matrix_set(Vm, i, j, gsl_matrix_get(V, i, j));
                out_stream << gsl_matrix_get(V, i, j);
            }
            out_stream << "\n";
        }


        size_t imaxU = nn, jmaxW = 3;
        gsl_matrix *UxW = gsl_matrix_alloc(imaxU, jmaxW);
        gsl_matrix *UxWxV = gsl_matrix_alloc(UxW->size1, Vm->size1);
        matrix_mul(UxW, Am, Wm);

        out_stream << "\nUxW\n";
        for(size_t i = 0; i < UxW->size1; ++ i)
        {
            for(size_t j = 0; j < UxW->size2; ++ j)
            {
                out_stream << gsl_matrix_get(UxW, i, j);
            }
            out_stream << "\n";
        }

        matrix_mul(UxWxV, UxW, Vm);

        out_stream << "\n\nRESULT\n\n";

        QString str[] = {"WikiLeaks", "арестова", "великобрита", "вручен", "нобелевск", "основател", "полиц", "прем", "прот", "стран", "суд", "сша", "церемон"};

        for(size_t i = 0; i < UxWxV->size1; ++ i)
        {
            out_stream << i << str[i];
            for(size_t j = 0; j < UxWxV->size2; ++ j)
            {
                out_stream << gsl_matrix_get(UxWxV, i, j);
            }
            out_stream << "\n";
        }


        out->close();
    }
    else
        qDebug() << "file !open.";


    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(work);
    qDebug() << "end";
    return a.exec();
}
