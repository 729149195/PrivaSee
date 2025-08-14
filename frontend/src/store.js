import { create } from 'zustand'

export const useStore = create((set, get) => ({
  // text state for TextTest component
  text: '',
  setText: (value) => set({ text: value }),
}))


