import { Card, CardContent, CardHeader } from '../ui/card';
import { Badge } from '../ui/badge';

export interface StyleGuideItem {
  category: string;
  guideNumber: number;
  title: string;
  description: string;
  examples: {
    correct?: string[];
    incorrect?: string[];
  };
}

interface StyleGuideCardProps {
  item: StyleGuideItem;
}

export function StyleGuideCard({ item }: StyleGuideCardProps) {
  return (
    <Card className="shadow-sm border-l-4 border-l-primary bg-white">
      <CardHeader className="pb-4">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-primary text-white">
              <span className="text-lg font-semibold">
                {item.guideNumber}
              </span>
            </div>
          </div>
          <div className="flex-1 space-y-2">
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className="text-xs text-[#717182]"
              >
                {item.category}
              </Badge>
            </div>
            <h3 className="leading-relaxed">
              <strong>Style Guide {item.guideNumber}</strong>
            </h3>
            <p className="text-sm leading-relaxed">- {item.title}</p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0 space-y-4">
        <div className="pl-16">
          <p className="text-sm leading-relaxed text-muted-foreground mb-4">
            {item.description}
          </p>

          {item.examples.correct && item.examples.correct.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm">
                <strong>Correct:</strong>
              </p>
              {item.examples.correct.map((example, i) => (
                <p key={i} className="text-sm pl-4 text-green-600">
                  {example}
                </p>
              ))}
            </div>
          )}

          {item.examples.incorrect && item.examples.incorrect.length > 0 && (
            <div className="space-y-2 mt-3">
              <p className="text-sm">
                <strong>Incorrect:</strong>
              </p>
              {item.examples.incorrect.map((example, i) => (
                <p key={i} className="text-sm pl-4 text-red-600">
                  {example}
                </p>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}