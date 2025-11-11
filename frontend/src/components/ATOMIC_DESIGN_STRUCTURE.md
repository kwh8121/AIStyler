# Atomic Design Structure

## Directory Structure
```
src/
├── components/
│   ├── atoms/           # Basic building blocks
│   │   ├── IconButton.tsx
│   │   ├── LoadingSpinner.tsx
│   │   ├── ProgressCircle.tsx
│   │   └── StatusBadge.tsx
│   │
│   ├── molecules/       # Combinations of atoms
│   │   ├── HeaderSection.tsx
│   │   ├── HistoryCard.tsx
│   │   ├── LoadingStep.tsx
│   │   ├── StyleGuideCard.tsx
│   │   └── TextEditor.tsx
│   │
│   ├── organisms/       # Complex UI sections
│   │   ├── AppHeader.tsx
│   │   ├── HistorySidebar.tsx
│   │   ├── LoadingModal.tsx
│   │   ├── StyleGuidePopup.tsx
│   │   └── TextEditorSection.tsx
│   │
│   ├── templates/       # Page layouts (to be added)
│   │
│   └── ui/              # Shadcn UI components (unchanged)
│
└── App.tsx              # Main application using organisms

```

## Component Hierarchy

### Atoms (Basic Building Blocks)
- **IconButton**: Reusable button with icon support
- **LoadingSpinner**: Animated loading indicator
- **ProgressCircle**: Circular progress indicator with percentage
- **StatusBadge**: Badge for displaying status types (AI, 번역, 복원)

### Molecules (Combinations of Atoms)
- **HeaderSection**: Logo + Title + Subtitle combination
- **HistoryCard**: Card displaying history item with status badge
- **LoadingStep**: Step indicator with icon and description
- **StyleGuideCard**: Card for displaying style guide rules
- **TextEditor**: Text area with label, icon, and optional actions

### Organisms (Complex UI Sections)
- **AppHeader**: Application header with branding and history button
- **HistorySidebar**: Sidebar containing history cards and actions
- **LoadingModal**: Modal with loading states (simple/progress variants)
- **StyleGuidePopup**: Popup displaying style guide cards
- **TextEditorSection**: Main editor section with input/output areas

## Key Benefits of This Structure

1. **Reusability**: Atoms and molecules can be reused across different organisms
2. **Maintainability**: Clear separation of concerns and component responsibilities
3. **Scalability**: Easy to add new components at any level
4. **Testability**: Smaller components are easier to test in isolation
5. **Consistency**: Shared atoms ensure consistent UI throughout the app

## Usage Example

```tsx
// App.tsx now uses clean, composed organisms
<AppHeader onHistoryClick={handleHistory} />
<TextEditorSection {...editorProps} />
<HistorySidebar {...sidebarProps} />
<LoadingModal {...modalProps} />
```

## Next Steps
- Add templates for different page layouts
- Create pages directory for route-specific components
- Add component documentation with Storybook
- Implement component testing