#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
using namespace std;
#include "cliente.h"

enum Escolhas {CRIAR=1, TELA, ARQUIVOTEXTO, ATUALIZAR, NOVO, APAGAR, FIM};

//------------------------------------------------------------------------
Escolhas enterChoice()
// pre: nenhuma
// pos: exibe menu na tela e solicita acao a ser tomada ao usuario
{
    int menuChoice;
    cout << "\nMenu:" << endl
         << "1 - cria registros vazios no arquivo\n"
         << "2 - lista os dados na tela\n"
         << "3 - armazena os dados no arquivo texto \"print.txt\"\n"
         << "4 - atualiza uma conta que ja contenha informacoes\n"
         << "5 - insere informacoes em uma nova conta\n"
         << "6 - apaga informacoes de uma conta\n"
         << "7 – fim do programa\n"
         << "Opcao: ";
    cin >> menuChoice;
    return (Escolhas) menuChoice;
}

//------------------------------------------------------------------------
void create(fstream &f)
// pre: arquivo f aberto
// pos: insere/sobrepoe 100 registros no arquivo, destruindo
//      qualquer informacao anterior
{
    cliente clienteVazio = {0, "", 0.0};

    f.seekg(0);
    for (int i = 0; i < 100; i++)
        f.write((const char *)(&clienteVazio), sizeof(cliente));
}

//------------------------------------------------------------------------
void outputLine(ostream &output, const cliente &c)
// pre: arquivo output aberto para escrita, cliente c contendo dados de cliente
// pos: escreve dados do cliente c no arquivo output
{
    output << setiosflags(ios::left)
           << setw(10) << c.numero
           << setw(30) << c.nome
           << setw(10) << setprecision(2) << resetiosflags(ios::left)
           << setiosflags(ios::fixed | ios::showpoint) << c.saldo << '\n';
}

//------------------------------------------------------------------------
void screen(fstream &f)
// pre: Arquivo f contendo dados de clientes aberto
// pos: Cria um arquivo texto formatado para impressao
{
    cliente c;

    cout << setiosflags(ios::left)
         << setw(10) << "Conta"
         << setw(30) << "Nome"
         << resetiosflags(ios::left) << setw(10) << "Saldo" << endl;

    f.seekg(0);
    f.read((char *)(&c), sizeof(cliente));
    while (!f.eof())
    {
        if (c.numero != 0)
            outputLine(cout, c);
        f.read((char *)(&c), sizeof(cliente));
    }
}

//------------------------------------------------------------------------
void textFile(fstream &f)
// pre: Arquivo f contendo dados de clientes aberto
// pos: Cria um arquivo texto formatado para impressao
{
    cliente c;
    ofstream outPrintFile("print.txt", ios::out);

    if (!outPrintFile)
    {
        cerr << "Arquivo print.txt nao pode ser aberto." << endl;
        exit(1);
    }

    outPrintFile << setiosflags(ios::left)
                 << setw(10) << "Conta"
                 << setw(30) << "Nome"
                 << resetiosflags(ios::left) << setw(10) << "Saldo" << endl;

    f.seekg(0);
    f.read((char *)(&c), sizeof(cliente));
    while (!f.eof())
    {
        if (c.numero != 0)
            outputLine(outPrintFile, c);
        f.read((char *)(&c), sizeof(cliente));
    }
    outPrintFile.close();
}

//------------------------------------------------------------------------
int getAccount(string msg)
// pre: nenhuma
// pos: imprime msg, obtem do teclado um numero de conta entre 1 e 100 e retorna-o
{
    int conta;

    do
    {
        cout << msg << " (1 - 100): ";
        cin >> conta;
    } while (conta < 1 || conta > 100);

    return conta;
}

//------------------------------------------------------------------------
void updateRecord(fstream &f)
// pre: Arquivo f aberto
// pos: Atualiza o saldo de uma conta
{
    int conta;
    cliente c;
    float transacao;

    conta = getAccount("Conta a ser atualizada");
    f.seekg((conta - 1) * sizeof(cliente)); // posicionar na conta desejada
    f.read((char *)(&c), sizeof(cliente));  // ler dados da conta
    if (c.numero != 0) // conta contem informacao?
    {
        outputLine(cout, c);
        cout << "\nEntre deposito (+) ou retirada (-): ";
        cin >> transacao;
        c.saldo += transacao;
        outputLine(cout, c);
        f.seekp((conta - 1) * sizeof(cliente));      // posicionar na conta desejada
        f.write((const char *)(&c), sizeof(cliente)); // atualizar
    }
    else
        cerr << "Conta #" << conta << " nao possui informacao." << endl;
}

//------------------------------------------------------------------------
void newRecord(fstream &f)
// pre: arquivo f aberto
// pos: insere informacao em um registro de conta que esteja vazio (sem informacao)
{
    int conta;
    cliente c;

    conta = getAccount("Numero da nova conta");
    f.seekg((conta - 1) * sizeof(cliente)); // posicionar na conta desejada
    f.read((char *)(&c), sizeof(cliente));  // ler dados da conta
    if (c.numero == 0)
    {
        cout << "Entre nome, saldo\n? ";
        cin >> c.nome >> c.saldo;
        c.numero = conta;
        f.seekp((conta - 1) * sizeof(cliente));      // posicionar na conta desejada
        f.write((const char *)(&c), sizeof(cliente)); // atualizar
    }
    else
        cerr << "Conta #" << conta << " ja possui informacao." << endl;
}

//------------------------------------------------------------------------
void deleteRecord(fstream &f)
// pre: arquivo f aberto
// pos: conta fornecida pelo usuario e apagada
//      (informação em branco)
{
    int conta;
    cliente c, clienteVazio = {0, "", 0.0};

    conta = getAccount("Conta a ser apagada");
    f.seekg((conta - 1) * sizeof(cliente));
    f.read((char *)(&c), sizeof(cliente));
    if (c.numero != 0)
    {
        f.seekp((conta - 1) * sizeof(cliente));
        f.write((const char *)(&clienteVazio), sizeof(cliente));
        cout << "Conta #" << conta << " apagada." << endl;
    }
    else
        cerr << "Conta #" << conta << " ja esta apagada." << endl;
}

//------------------------------------------------------------------------
int main()
{
    Escolhas opcao;
    fstream inOutCredito("credito.dat", ios::in | ios::out); // leitura e escrita

    if (!inOutCredito)
    {
        cerr << "Arquivo credito.dat nao pode ser aberto." << endl;
        exit(1);
    }

    while ((opcao = enterChoice()) != FIM)
    {
        switch (opcao)
        {
        case CRIAR:
            create(inOutCredito);
            break;
        case TELA:
            screen(inOutCredito);
            break;
        case ARQUIVOTEXTO:
            textFile(inOutCredito);
            break;
        case ATUALIZAR:
            updateRecord(inOutCredito);
            break;
        case NOVO:
            newRecord(inOutCredito);
            break;
        case APAGAR:
            deleteRecord(inOutCredito);
            break;
        default:
            cerr << "Opcao incorreta\n";
            break;
        }
        inOutCredito.clear(); // limpa o indicador de final-de arquivo
    }
    return 0;
}
