# PRD — CNN Lab TCC

## Contexto

Ferramenta pessoal para experimentar otimização de hiperparâmetros de CNNs.
É o precursor prático do experimento de TCC, que compara algoritmo genético
contra Bayesian Optimization e Random Search sob mesmo orçamento de avaliações.

Esta POC entrega apenas o algoritmo genético. BO e Random Search entram depois,
mas a arquitetura já reserva lugar para eles.

## Problema

Ajustar hiperparâmetros de CNN na mão é lento e não deixa rastro. Cada tentativa
vira um número solto num terminal que some quando você fecha a janela. É preciso
poder: disparar buscas, acompanhar ao vivo, deixar rodando enquanto dorme, e
consultar tudo depois.

## Usuário

Um usuário, dono da máquina, rodando local no Windows com uma RTX 5070 (12 GB).
Não há multiusuário, autenticação, rede, ou deploy.

## Objetivos

1. Disparar um treino clássico com hiperparâmetros escolhidos à mão.
2. Disparar uma sessão de AG que busca hiperparâmetros automaticamente.
3. Acompanhar progresso ao vivo, com barra e métricas.
4. Modo Sleep: enfileirar N sessões de AG e deixar rodando sem supervisão.
5. Consultar histórico de execuções, separado por tipo.
6. Carregar uma rede salva e classificar imagens (do dataset ou próprias).

## Fora de escopo nesta POC

Estes itens são explicitamente adiados. Implementá-los é
um erro de escopo.

- Bayesian Optimization e Random Search
- Múltiplas arquiteturas de CNN selecionáveis (arquitetura é fixa e parametrizada)
- Arquitetura no genoma (o genoma otimiza treino + largura, não topologia livre)
- Salvar redes intermediárias de gerações anteriores (só a melhor final)
- Cross-validation com k folds (usa split fixo train/val)
- Múltiplas seeds por configuração
- Datasets além de CIFAR-100
- Temas customizáveis, perfis, ASCII art, telas de instalação
- Empacotamento, instalador, distribuição
- Testes de UI

## Critérios de aceite

A POC está pronta quando, numa máquina limpa:

- [ ] `cnnlab doctor` reporta GPU detectada e dataset presente
- [ ] `cnnlab ga --config exemplo.json` roda uma sessão inteira sem TUI e grava em `runs/`
- [ ] Na TUI, é possível ajustar parâmetros na sidebar e iniciar um treino clássico
- [ ] Na TUI, é possível iniciar uma sessão de AG e ver geração/indivíduo/época avançando ao vivo
- [ ] O botão de parar interrompe em até uma época
- [ ] Modo Sleep aceita N sessões e as executa em sequência sem intervenção
- [ ] Fechar e reabrir a TUI mostra as execuções anteriores no histórico
- [ ] Matar o processo no meio de uma run e reabrir não corrompe o histórico
- [ ] É possível carregar a melhor rede de uma run e classificar uma imagem
- [ ] Uma sessão de AG com população 8 e 6 gerações termina em menos de 40 minutos

## Meta de demonstração

Mostrar 3 sessões de AG de 5 a 6 gerações cada. A acurácia final não é o
critério — o critério é o pipeline funcionar de ponta a ponta e a curva de
melhor fitness por geração subir. Uma CNN básica em CIFAR-100 com poucas épocas
vai ficar numa acurácia baixa, e isso é esperado.

## Riscos conhecidos

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| DEAP pode não ter wheel para Python 3.14 | Bloqueia o AG | Fallback: AG próprio em ~80 linhas, spec em `docs/06` |
| CIFAR-100 com CNN pequena converge mal | Demonstração parece fraca | Comunicar que a métrica é a curva, não o valor absoluto |
| Crash da TUI de madrugada | Perde a noite | Persistência a cada indivíduo, `queue.json` retomável |
| Windows suspender | Perde a noite | `powercfg /change standby-timeout-ac 0` |
| OOM de VRAM com batch grande | Run falha | Clamp de batch_size no espaço de busca, try/except com retry menor |
