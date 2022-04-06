package vggface2

import (
	"fmt"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type ConvBlockModule struct {
	model   *godl.Model
	layer   godl.LayerType
	opts    BlockOpts
	bns     []*godl.BatchNormModule
	weights godl.Nodes
}

func (m *ConvBlockModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]
	{
		x = gorgonia.Must(gorgonia.Conv2d(x, m.weights[0], tensor.Shape{1, 1}, []int{0, 0}, m.opts.Stride, []int{1, 1}))

		result := m.bns[0].Forward(x)

		x = gorgonia.Must(gorgonia.Rectify(result[0]))
	}

	{
		x = gorgonia.Must(gorgonia.Conv2d(x, m.weights[1], m.opts.KernelSize, []int{1, 1}, []int{1, 1}, []int{1, 1}))

		result := m.bns[1].Forward(x)

		x = gorgonia.Must(gorgonia.Rectify(result[0]))
	}

	{
		x = gorgonia.Must(gorgonia.Conv2d(x, m.weights[2], tensor.Shape{1, 1}, []int{0, 0}, []int{1, 1}, []int{1, 1}))

		result := m.bns[2].Forward(x)

		x = result[0]
	}

	{
		shortCut := gorgonia.Must(gorgonia.Conv2d(x, m.weights[3], tensor.Shape{1, 1}, []int{0, 0}, m.opts.Stride, []int{1, 1}))

		result := m.bns[3].Forward(shortCut)

		x = gorgonia.Must(gorgonia.Add(x, result[0]))
		x = gorgonia.Must(gorgonia.Rectify(x))
	}

	return godl.Nodes{x}
}

func ConvBlock(m *godl.Model, opts BlockOpts) *ConvBlockModule {
	lt := godl.AddLayer("vggface2.ConvBlock")

	conv1ReduceName := fmt.Sprintf("conv%d_%d_1x1_reduce", opts.Stage, opts.Block)
	conv1IncreaseName := fmt.Sprintf("conv%d_%d_1x1_increase", opts.Stage, opts.Block)
	conv1ProjName := fmt.Sprintf("conv%d_%d_1x1_proj", opts.Stage, opts.Block)
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

	bn4 := godl.BatchNorm2d(m, godl.BatchNormOpts{
		InputSize: opts.Filters[2],
		ScaleName: conv1ProjName + "/bn/gamma",
		BiasName:  conv1ProjName + "/bn/beta",
	})
	w4 := m.AddWeights(lt, tensor.Shape{opts.Filters[2], opts.Filters[1], 3, 3}, godl.NewWeightsOpts{
		UniqueName: conv1ProjName + "/kernel",
	})

	return &ConvBlockModule{
		model: m,
		layer: lt,
		opts:  opts,
		bns: []*godl.BatchNormModule{
			bn1, bn2, bn3, bn4,
		},
		weights: godl.Nodes{
			w1, w2, w3, w4,
		},
	}
}
