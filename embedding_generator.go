package godl

import (
	"sort"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type EmbeddingGeneratorModule struct {
	model         *Model
	opts          EmbeddingOpts
	skipEmbedding bool

	categoricalColumnIndexes []bool
	embeddings               []*EmbeddingModule
}

func (m *EmbeddingGeneratorModule) Forward(inputs ...*Node) Nodes {
	err := m.model.CheckArity("EmbeddingGenerator", inputs, 1)
	if err != nil {
		panic(err)
	}

	x := inputs[0]
	if m.skipEmbedding {
		return Nodes{x}
	}

	cols := make([]*gorgonia.Node, len(m.categoricalColumnIndexes))
	catFeatCounter := 0

	for featInitIdx, isCategorical := range m.categoricalColumnIndexes {
		s := gorgonia.Must(gorgonia.Slice(x, nil, gorgonia.S(featInitIdx)))

		if isCategorical {
			s := gorgonia.Must(gorgonia.ConvType(s, tensor.Float32, tensor.Int))
			result := m.embeddings[catFeatCounter].Forward(s)

			cols[featInitIdx] = result[0]

			catFeatCounter++
		} else {
			cols[featInitIdx] = gorgonia.Must(gorgonia.Reshape(s, tensor.Shape{s.Shape().TotalSize(), 1}))
		}
	}

	output := gorgonia.Must(gorgonia.Concat(1, cols...))

	return Nodes{output}
}

func EmbeddingGenerator(m *Model, inputDims int, catDims []int, catIdxs []int, catEmbDim []int, opts EmbeddingOpts) *EmbeddingGeneratorModule {
	skipEmbedding := false
	if len(catDims) == 0 || len(catIdxs) == 0 {
		skipEmbedding = true
	}

	sort.Slice(catIdxs, func(i, j int) bool {
		return catIdxs[i] < catIdxs[j]
	})

	embeddings := make([]*EmbeddingModule, len(catIdxs))
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

	return &EmbeddingGeneratorModule{
		model:                    m,
		opts:                     opts,
		skipEmbedding:            skipEmbedding,
		categoricalColumnIndexes: categoricalColumnIndexes,
		embeddings:               embeddings,
	}
}
