# Sistema Inteligente de Contagem de Carros em Semáforo

![Interface](imgs/img1.png)

Este repositório contém o código do projeto de um sistema inteligente para contagem de veículos em tempo real em um semáforo. 

 
- **Vídeos utilizados:** [Friantroulette YouTube Channel](https://www.youtube.com/@friantroulette)

## Estrutura de Diretórios

```plaintext
├── data/                  
│   ├── __init__.py
│   └── data_manager.py    
├── gui/                   # Interface gráfica
│   ├── __init__.py
│   ├── app.py             
│   └── gui.py             
├── track/                 # Rastreamento de veículos
│   ├── __init__.py
│   └── tracker.py        
├── models/                # Pesos do modelo treinado
│   └── model_main.pt           # <-- Coloque aqui o peso do modelo
├── reports/               # Saídas e relatórios
│   ├── report.csv         # Exemplo de relatório em CSV
│   └── report.png         # Exemplo de relatório em imagem
├── main.py                # Ponto de entrada da aplicação
└── README.md              
```  

## Pré-requisitos  

- Python 3.7 ou superior  
- pip  

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Download dos Pesos do Modelo

Antes de executar o sistema, faça o download dos pesos do modelo treinado através do link abaixo:

[Download do modelo treinado](https://drive.google.com/file/d/12CviN5DbXcsvD0IInRRMEFROtc0dFAB5/view?usp=sharing)

O arquivo deve ficar em models/model_main.py

## Uso  

Execute a aplicação completa (GUI + contagem):

```bash
python main.py
```
# Componentes e lógica da interface

1. **data_manager.py**: carrega e pré-processa os frames dos vídeos  
2. **tracker.py**: executa o rastreamento de veículos objeto a objeto  
3. **gui.py / app.py**: apresenta interface em tempo real com contagem atualizada  
4. **main.py**: integra todos os módulos e inicia a aplicação



## Contato  

Entre em contato pelo e-mail: jglg@cin.ufpe.br



