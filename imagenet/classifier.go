package imagenet

import (
	"image"
	"sync"

	"github.com/dcu/godl"
	"github.com/dcu/godl/imageutils"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Classifier is an imagenet classifier
type Classifier struct {
	m *godl.Model
	x *gorgonia.Node

	output gorgonia.Value
	mutex  sync.Mutex
}

func NewClassifier(builder func(m *godl.Model) godl.Module, width, height int) *Classifier {
	m := godl.NewModel()
	module := builder(m)
	x := gorgonia.NewTensor(m.TrainGraph(), tensor.Float32, 4, gorgonia.WithShape(1, 3, width, height), gorgonia.WithName("x"))

	result := module.Forward(x)
	c := &Classifier{
		m: m,
		x: x,
	}

	gorgonia.Read(result[0], &c.output)

	return c
}

func (c *Classifier) Model() *godl.Model {
	return c.m
}

func (c *Classifier) Predict(img image.Image) (string, float64, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	input := imageutils.ToTensor(img, imageutils.ToTensorOpts{})

	err := gorgonia.Let(c.x, input)
	if err != nil {
		return "", 0.0, err
	}

	err = c.m.Run()
	if err != nil {
		return "", 0.0, err
	}

	outputTensor := c.output.(tensor.Tensor)
	max, err := tensor.Argmax(outputTensor, 1)
	if err != nil {
		return "", 0.0, err
	}

	index := max.Data().([]int)[0]

	val, err := outputTensor.At(0, index)
	if err != nil {
		return "", 0.0, err
	}

	return Labels[index], val.(float64), nil
}
