package storage

import (
	"fmt"
	"log"
	"strings"

	"gorgonia.org/tensor"
)

// Storage is in charge of loading the weights from files
type Storage struct {
	Cost       float64
	Learnables map[string]Weight
}

// NewStorage instantiates a storage
func NewStorage() *Storage {
	return &Storage{
		Cost:       0.0,
		Learnables: map[string]Weight{},
	}
}

// TensorByName returns the tensor associated to a weight name
func (l *Storage) TensorByName(name string) (tensor.Tensor, error) {
	t, ok := l.Learnables[name]
	if !ok {
		return nil, ErrLearnableNotFound
	}

	return t.Value.(tensor.Tensor), nil
}

// Load loads the weights in the given path
func (l *Storage) LoadFile(filePath string) error {
	if strings.Contains(filePath, ".nn1") {
		return LoadNN1(l, filePath)
	} else {
		return fmt.Errorf("extension %v is not supported yet", filePath)
	}
}

// AddWeights adds weights to the storage
func (l *Storage) AddWeights(weights ...Weight) {
	for _, w := range weights {
		if _, ok := l.Learnables[w.Name]; ok {
			log.Panicf("weight %s is already present in the storage", w.Name)
		}

		l.Learnables[w.Name] = w
	}
}
