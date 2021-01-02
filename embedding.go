package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type EmbeddingOpts struct {
	WeightsInit gorgonia.InitWFn
}

// Embedding implements a embedding layer
func (m *Model) Embedding(embeddingSize int, embeddingDim int, opts EmbeddingOpts) Layer {
	w := m.addWeights(tensor.Shape{embeddingSize, embeddingDim}, opts.WeightsInit)
	oneHots := make(gorgonia.Nodes, embeddingSize)

	for i := 0; i < embeddingSize; i++ {
		oh := buildOneHotAt(i, embeddingSize)
		oneHots[i] = gorgonia.NewMatrix(m.g, tensor.Float64, gorgonia.WithValue(oh), gorgonia.WithName(fmt.Sprintf("one-hot.%d.%d", learnablesCount, i)))
	}

	return func(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
		err := m.checkArity("Embedding", inputs, 1)
		if err != nil {
			return nil, err
		}

		x := inputs[0]
		indexes := x.Value().Data().([]float64)
		nodes := make(gorgonia.Nodes, len(indexes))

		for i, index := range indexes {
			nodes[i] = gorgonia.Must(gorgonia.Mul(oneHots[int(index)], w))
		}

		result := gorgonia.Must(gorgonia.Concat(0, nodes...))

		return gorgonia.Reshape(result, append(x.Shape(), embeddingDim))
	}
}

func buildOneHotAt(index int, max int) tensor.Tensor {
	oneHotBacking := make([]float64, max)
	oneHotBacking[int(index)] = 1.0

	return tensor.New(
		tensor.WithShape(1, max),
		tensor.WithBacking(oneHotBacking),
	)
}
