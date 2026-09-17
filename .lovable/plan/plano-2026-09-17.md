# Plano

## O que vou mudar
- Trocar a frase padrão “o gato bebeu leite” por “A Anta comeu banana” em todos os pontos da experiência.
- Ajustar os exemplos explicativos que citam “gato”, “bebeu” e “leite” para usar “anta”, “comeu” e “banana”.
- Manter a simulação funcionando do mesmo jeito, apenas com os novos tokens e valores padrão correspondentes.

## Positional Encoding
- Inserir uma nova etapa logo depois de Embeddings e antes de Query, Key e Value.
- Mostrar a fórmula didática com seno e cosseno:
  - PE(pos, 2i) = sin(pos / 10000^(2i/d))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
- Exibir visualmente onde o positional encoding começa e termina.
- Mostrar a soma X + PE como a representação que segue para Q/K/V.

## Ajustes na navegação
- Atualizar a navegação lateral, barra de progresso e contagem de etapas para incluir a nova etapa.
- Atualizar o fluxo dos dados para ficar: Tokens → Embeddings → Positional Encoding → Q/K/V → demais etapas.

## Detalhes técnicos
- A matemática da atenção continuará recalculando no navegador.
- O pipeline passará a usar X com positional encoding somado antes das projeções Q, K e V.
- A simulação interativa também exibirá X, PE, X + PE e depois Q/K/V.
- Validarei no preview que a frase nova aparece, a nova etapa abre corretamente e não há erros visíveis.
