package godl

import (
	"sort"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func EmbeddingGenerator(m *Model, inputDims int, catDims []int, catIdxs []int, catEmbDim []int, opts EmbeddingOpts) Layer {
	skipEmbedding := false
	if len(catDims) == 0 || len(catIdxs) == 0 {
		skipEmbedding = true
	}

	sort.Slice(catIdxs, func(i, j int) bool {
		return catIdxs[i] < catIdxs[j]
	})

	embeddings := make([]Layer, len(catIdxs))
	categoricalColumnIndexes := make([]bool, inputDims)

	for i, v := range catIdxs {
		embeddings[i] = Embedding(
			m,
			catDims[i],
			catEmbDim[i],
			opts,
		)

		categoricalColumnIndexes[v] = true
	}

	return func(inputs ...*gorgonia.Node) (Result, error) {
		err := m.CheckArity("EmbeddingGenerator", inputs, 1)
		if err != nil {
			return Result{}, err
		}

		x := inputs[0]
		if skipEmbedding {
			return Result{Output: x}, nil
		}

		cols := make([]*gorgonia.Node, len(categoricalColumnIndexes))
		catFeatCounter := 0

		for featInitIdx, isCategorical := range categoricalColumnIndexes {
			s := gorgonia.Must(gorgonia.Slice(x, nil, gorgonia.S(featInitIdx)))

			if isCategorical {
				s := gorgonia.Must(gorgonia.ConvType(s, tensor.Float32, tensor.Int))
				result, err := embeddings[catFeatCounter](s)
				if err != nil {
					panic(err)
				}

				cols[featInitIdx] = result.Output

				catFeatCounter++
			} else {
				cols[featInitIdx] = gorgonia.Must(gorgonia.Reshape(s, tensor.Shape{s.Shape().TotalSize(), 1}))
			}
		}

		output := gorgonia.Must(gorgonia.Concat(1, cols...))

		return Result{Output: output}, nil
	}
}
