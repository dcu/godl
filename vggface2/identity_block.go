package vggface2

import (
	"fmt"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type BlockOpts struct {
	Filters    [3]int
	Stride     []int
	Stage      int
	KernelSize tensor.Shape
	Block      int
}

type IdentityBlockModule struct {
	model   *godl.Model
	layer   godl.LayerType
	opts    BlockOpts
	bns     []*godl.BatchNormModule
	weights []*godl.Node
}

func (m *IdentityBlockModule) Name() string {
	return "IdentityBlock"
}

func (m *IdentityBlockModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]
	{

		x = gorgonia.Must(gorgonia.Conv2d(x, m.weights[0], tensor.Shape{1, 1}, []int{0, 0}, []int{1, 1}, []int{1, 1}))

		result := m.bns[0].Forward(x)
		x = gorgonia.Must(gorgonia.Rectify(result[0]))
	}

	{

		x = gorgonia.Must(gorgonia.Conv2d(x, m.weights[1], m.opts.KernelSize, []int{0, 0}, []int{1, 1}, []int{1, 1}))
		result := m.bns[1].Forward(x)

		x = gorgonia.Must(gorgonia.Rectify(result[0]))
	}

	{

		x = gorgonia.Must(gorgonia.Conv2d(x, m.weights[2], tensor.Shape{1, 1}, []int{0, 0}, []int{1, 1}, []int{1, 1}))

		result := m.bns[2].Forward(x)

		x = gorgonia.Must(gorgonia.Add(result[0], inputs[0]))
		x = gorgonia.Must(gorgonia.Rectify(x))
	}

	return godl.Nodes{x}
}

func IdentityBlock(m *godl.Model, opts BlockOpts) *IdentityBlockModule {
	lt := godl.AddLayer("vggface2.IdentityBlock")

	conv1ReduceName := fmt.Sprintf("conv%d_%d_1x1_reduce", opts.Stage, opts.Block)
	conv1IncreaseName := fmt.Sprintf("conv%d_%d_1x1_increase", opts.Stage, opts.Block)
	conv3Name := fmt.Sprintf("conv%d_%d_3x3", opts.Stage, opts.Block)

	bn1 := godl.BatchNorm2d(m, godl.BatchNormOpts{
		InputSize: opts.Filters[0],
		ScaleName: conv1ReduceName + "/bn/gamma",
		BiasName:  conv1ReduceName + "/bn/beta",
	})
	w1 := m.AddWeights(lt, tensor.Shape{opts.Filters[0], 3, 3, 3}, godl.NewWeightsOpts{
		UniqueName: conv1ReduceName + "/kernel",
	})

	bn2 := godl.BatchNorm2d(m, godl.BatchNormOpts{
		InputSize: opts.Filters[1],
		ScaleName: conv3Name + "/bn/gamma",
		BiasName:  conv3Name + "/bn/beta",
	})
	w2 := m.AddWeights(lt, tensor.Shape{opts.Filters[1], opts.Filters[0], 3, 3}, godl.NewWeightsOpts{
		UniqueName: conv3Name + "/kernel",
	})

	bn3 := godl.BatchNorm2d(m, godl.BatchNormOpts{
		InputSize: opts.Filters[2],
		ScaleName: conv1IncreaseName + "/bn/gamma",
		BiasName:  conv1IncreaseName + "/bn/beta",
	})
	w3 := m.AddWeights(lt, tensor.Shape{opts.Filters[2], opts.Filters[1], 3, 3}, godl.NewWeightsOpts{
		UniqueName: conv1IncreaseName + "/kernel",
	})

	return &IdentityBlockModule{
		model:   m,
		layer:   lt,
		opts:    opts,
		bns:     []*godl.BatchNormModule{bn1, bn2, bn3},
		weights: []*godl.Node{w1, w2, w3},
	}
}
