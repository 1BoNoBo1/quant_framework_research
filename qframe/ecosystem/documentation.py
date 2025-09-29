"""
QFrame Documentation Portal
==========================

Interactive documentation system with tutorials, code examples,
and best practices for the quantitative trading framework.
"""

import ast
import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from uuid import uuid4

import markdown
import pandas as pd
from jinja2 import Template, Environment, FileSystemLoader
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

from qframe.core.container import injectable
from qframe.core.config import FrameworkConfig


class DocumentationType(str, Enum):
    """Types of documentation content."""
    TUTORIAL = "tutorial"
    GUIDE = "guide"
    REFERENCE = "reference"
    API_DOC = "api_doc"
    EXAMPLE = "example"
    BEST_PRACTICE = "best_practice"
    FAQ = "faq"
    CHANGELOG = "changelog"


class DifficultyLevel(str, Enum):
    """Content difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentStatus(str, Enum):
    """Documentation content status."""
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass
class CodeExample:
    """Interactive code example with execution capabilities."""

    # Core properties
    title: str
    description: str
    code: str
    language: str = "python"

    # Metadata
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    # Execution properties
    is_executable: bool = True
    expected_output: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None

    # Documentation properties
    explanation: str = ""
    related_examples: List[str] = field(default_factory=list)
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER

    def execute(self) -> Dict[str, Any]:
        """Execute the code example and return results."""
        try:
            # Setup namespace
            namespace = {}

            # Execute setup code if provided
            if self.setup_code:
                exec(self.setup_code, namespace)

            # Execute main code
            exec(self.code, namespace)

            # Extract results
            output = namespace.get('__output__', 'Code executed successfully')

            return {
                'success': True,
                'output': output,
                'namespace': {k: str(v) for k, v in namespace.items() if not k.startswith('_')}
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

        finally:
            # Execute teardown code if provided
            if self.teardown_code:
                try:
                    exec(self.teardown_code, namespace)
                except Exception:
                    pass  # Ignore teardown errors

    def to_html(self) -> str:
        """Convert code example to HTML with syntax highlighting."""
        lexer = get_lexer_by_name(self.language)
        formatter = HtmlFormatter(style='default', cssclass='highlight')

        highlighted_code = highlight(self.code, lexer, formatter)

        return f"""
        <div class="code-example" data-example-id="{uuid4().hex}">
            <div class="example-header">
                <h3>{self.title}</h3>
                <span class="difficulty-badge {self.difficulty.value}">{self.difficulty.value.title()}</span>
            </div>
            <div class="example-description">
                <p>{self.description}</p>
            </div>
            <div class="example-code">
                {highlighted_code}
            </div>
            <div class="example-actions">
                <button class="run-code-btn" onclick="runExample(this)">Run Code</button>
                <button class="copy-code-btn" onclick="copyCode(this)">Copy</button>
            </div>
            <div class="example-output" style="display: none;">
                <h4>Output:</h4>
                <pre class="output-content"></pre>
            </div>
            {f'<div class="example-explanation"><h4>Explanation:</h4><p>{self.explanation}</p></div>' if self.explanation else ''}
        </div>
        """

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'description': self.description,
            'code': self.code,
            'language': self.language,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags,
            'is_executable': self.is_executable,
            'expected_output': self.expected_output,
            'dependencies': self.dependencies,
            'setup_code': self.setup_code,
            'teardown_code': self.teardown_code,
            'explanation': self.explanation,
            'related_examples': self.related_examples,
            'difficulty': self.difficulty.value
        }


@dataclass
class InteractiveTutorial:
    """Interactive tutorial with step-by-step progression."""

    # Core properties
    title: str
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    # Tutorial properties
    difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    estimated_duration: int = 30  # minutes
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)

    # Progress tracking
    completion_rate: float = 0.0
    user_progress: Dict[str, Any] = field(default_factory=dict)

    # Content organization
    tags: List[str] = field(default_factory=list)
    category: str = ""
    status: ContentStatus = ContentStatus.DRAFT

    def add_step(self,
                 title: str,
                 content: str,
                 code_example: Optional[CodeExample] = None,
                 quiz_questions: Optional[List[Dict[str, Any]]] = None,
                 resources: Optional[List[str]] = None) -> None:
        """Add a step to the tutorial."""
        step = {
            'id': len(self.steps) + 1,
            'title': title,
            'content': content,
            'code_example': code_example.to_dict() if code_example else None,
            'quiz_questions': quiz_questions or [],
            'resources': resources or [],
            'completed': False
        }
        self.steps.append(step)

    def mark_step_completed(self, step_id: int, user_id: str) -> None:
        """Mark a step as completed for a user."""
        if step_id <= len(self.steps):
            if user_id not in self.user_progress:
                self.user_progress[user_id] = {'completed_steps': set()}

            self.user_progress[user_id]['completed_steps'].add(step_id)
            self._update_completion_rate(user_id)

    def get_next_step(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the next uncompleted step for a user."""
        user_data = self.user_progress.get(user_id, {'completed_steps': set()})
        completed_steps = user_data['completed_steps']

        for step in self.steps:
            if step['id'] not in completed_steps:
                return step

        return None  # All steps completed

    def _update_completion_rate(self, user_id: str) -> None:
        """Update completion rate for a user."""
        user_data = self.user_progress[user_id]
        completed_count = len(user_data['completed_steps'])
        total_steps = len(self.steps)

        if total_steps > 0:
            user_data['completion_rate'] = completed_count / total_steps
            self.completion_rate = user_data['completion_rate']

    def to_html(self) -> str:
        """Convert tutorial to HTML format."""
        html_steps = []

        for i, step in enumerate(self.steps):
            step_html = f"""
            <div class="tutorial-step" data-step-id="{step['id']}">
                <div class="step-header">
                    <h3>Step {step['id']}: {step['title']}</h3>
                    <div class="step-progress">
                        <span class="step-number">{i + 1} of {len(self.steps)}</span>
                    </div>
                </div>
                <div class="step-content">
                    {markdown.markdown(step['content'])}
                </div>
            """

            if step['code_example']:
                example = CodeExample(**step['code_example'])
                step_html += f"<div class='step-code'>{example.to_html()}</div>"

            if step['quiz_questions']:
                step_html += "<div class='step-quiz'>"
                for j, question in enumerate(step['quiz_questions']):
                    step_html += f"""
                    <div class="quiz-question">
                        <h4>Question {j + 1}: {question['question']}</h4>
                        <div class="quiz-options">
                    """
                    for k, option in enumerate(question.get('options', [])):
                        step_html += f"""
                        <label>
                            <input type="radio" name="q{j}" value="{k}">
                            {option}
                        </label>
                        """
                    step_html += "</div></div>"
                step_html += "</div>"

            step_html += """
                <div class="step-actions">
                    <button class="complete-step-btn" onclick="completeStep(this)">Mark Complete</button>
                    <button class="next-step-btn" onclick="nextStep(this)">Next Step</button>
                </div>
            </div>
            """
            html_steps.append(step_html)

        return f"""
        <div class="interactive-tutorial" data-tutorial-id="{uuid4().hex}">
            <div class="tutorial-header">
                <h1>{self.title}</h1>
                <div class="tutorial-meta">
                    <span class="difficulty {self.difficulty.value}">{self.difficulty.value.title()}</span>
                    <span class="duration">{self.estimated_duration} minutes</span>
                    <span class="completion-rate">{self.completion_rate:.0%} complete</span>
                </div>
                <p class="tutorial-description">{self.description}</p>
            </div>
            <div class="tutorial-steps">
                {''.join(html_steps)}
            </div>
        </div>
        """

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'description': self.description,
            'steps': self.steps,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'difficulty': self.difficulty.value,
            'estimated_duration': self.estimated_duration,
            'prerequisites': self.prerequisites,
            'learning_objectives': self.learning_objectives,
            'completion_rate': self.completion_rate,
            'tags': self.tags,
            'category': self.category,
            'status': self.status.value
        }


@dataclass
class BestPractices:
    """Collection of best practices and guidelines."""

    # Core properties
    title: str
    category: str
    practices: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Organization
    tags: List[str] = field(default_factory=list)
    priority: int = 1  # 1=highest, 5=lowest
    applies_to: List[str] = field(default_factory=list)  # Components this applies to

    def add_practice(self,
                    title: str,
                    description: str,
                    rationale: str,
                    example: Optional[CodeExample] = None,
                    antipattern: Optional[CodeExample] = None,
                    importance: str = "medium") -> None:
        """Add a best practice."""
        practice = {
            'id': len(self.practices) + 1,
            'title': title,
            'description': description,
            'rationale': rationale,
            'example': example.to_dict() if example else None,
            'antipattern': antipattern.to_dict() if antipattern else None,
            'importance': importance,
            'created_at': datetime.now().isoformat()
        }
        self.practices.append(practice)

    def get_practices_by_importance(self, importance: str) -> List[Dict[str, Any]]:
        """Get practices filtered by importance level."""
        return [p for p in self.practices if p['importance'] == importance]

    def to_html(self) -> str:
        """Convert best practices to HTML format."""
        practices_html = []

        for practice in self.practices:
            practice_html = f"""
            <div class="best-practice" data-practice-id="{practice['id']}">
                <div class="practice-header">
                    <h3>{practice['title']}</h3>
                    <span class="importance-badge {practice['importance']}">{practice['importance'].title()}</span>
                </div>
                <div class="practice-description">
                    <p>{practice['description']}</p>
                </div>
                <div class="practice-rationale">
                    <h4>Why this matters:</h4>
                    <p>{practice['rationale']}</p>
                </div>
            """

            if practice['example']:
                example = CodeExample(**practice['example'])
                practice_html += f"""
                <div class="practice-example">
                    <h4>✅ Good Example:</h4>
                    {example.to_html()}
                </div>
                """

            if practice['antipattern']:
                antipattern = CodeExample(**practice['antipattern'])
                practice_html += f"""
                <div class="practice-antipattern">
                    <h4>❌ Avoid This:</h4>
                    {antipattern.to_html()}
                </div>
                """

            practice_html += "</div>"
            practices_html.append(practice_html)

        return f"""
        <div class="best-practices-collection">
            <div class="collection-header">
                <h1>{self.title}</h1>
                <div class="collection-meta">
                    <span class="category">{self.category}</span>
                    <span class="priority">Priority: {self.priority}</span>
                    <span class="practice-count">{len(self.practices)} practices</span>
                </div>
            </div>
            <div class="practices-list">
                {''.join(practices_html)}
            </div>
        </div>
        """

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'category': self.category,
            'practices': self.practices,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags,
            'priority': self.priority,
            'applies_to': self.applies_to
        }


@injectable
class DocumentationPortal:
    """Main documentation portal management system."""

    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.base_path = Path(getattr(config, 'docs_directory', './docs'))
        self.base_path.mkdir(exist_ok=True)

        # Content storage
        self.tutorials: Dict[str, InteractiveTutorial] = {}
        self.examples: Dict[str, CodeExample] = {}
        self.best_practices: Dict[str, BestPractices] = {}
        self.content_index: Dict[str, Dict[str, Any]] = {}

        # Template engine
        self.env = Environment(
            loader=FileSystemLoader(self.base_path / 'templates' if (self.base_path / 'templates').exists() else [])
        )

        # Initialize default content
        self._initialize_default_content()

    def _initialize_default_content(self) -> None:
        """Initialize default documentation content."""
        # Quick Start Tutorial
        quick_start = InteractiveTutorial(
            title="QFrame Quick Start Guide",
            description="Get started with QFrame quantitative trading framework",
            difficulty=DifficultyLevel.BEGINNER,
            estimated_duration=45,
            learning_objectives=[
                "Install and configure QFrame",
                "Create your first trading strategy",
                "Run a basic backtest",
                "Understand the framework architecture"
            ]
        )

        # Add steps to quick start
        quick_start.add_step(
            title="Installation",
            content="""
# Installing QFrame

QFrame can be installed using Poetry for dependency management:

```bash
git clone https://github.com/qframe/framework.git
cd qframe
poetry install
```

This will install all required dependencies and set up the development environment.
            """,
            code_example=CodeExample(
                title="Verify Installation",
                description="Check that QFrame is properly installed",
                code="""
import qframe
from qframe.core.config import FrameworkConfig

config = FrameworkConfig()
print(f"QFrame version: {qframe.__version__}")
print(f"Environment: {config.environment}")
""",
                expected_output="QFrame version: 1.0.0\nEnvironment: development"
            )
        )

        quick_start.add_step(
            title="Basic Configuration",
            content="""
# Framework Configuration

QFrame uses Pydantic for type-safe configuration. You can customize the framework
behavior through environment variables or configuration files.
            """,
            code_example=CodeExample(
                title="Basic Configuration Setup",
                description="Configure QFrame for your environment",
                code="""
from qframe.core.config import FrameworkConfig
from qframe.core.container import get_container

# Load configuration
config = FrameworkConfig()

# Initialize dependency injection container
container = get_container()
container.register_singleton(FrameworkConfig, lambda: config)

print(f"Database URL: {config.database.url}")
print(f"Redis URL: {config.redis.url}")
""",
                explanation="This sets up the basic configuration and dependency injection container."
            )
        )

        self.add_tutorial(quick_start)

        # Strategy Development Best Practices
        strategy_practices = BestPractices(
            title="Trading Strategy Development",
            category="Development",
            priority=1,
            applies_to=["strategies", "backtesting", "risk_management"]
        )

        strategy_practices.add_practice(
            title="Always Use Type Hints",
            description="Use Python type hints for all function parameters and return values",
            rationale="Type hints improve code readability, enable better IDE support, and help catch errors early",
            importance="high",
            example=CodeExample(
                title="Type Hints Example",
                description="Proper use of type hints in strategy development",
                code="""
from typing import List, Optional
import pandas as pd
from qframe.core.interfaces import Strategy, Signal

class MyStrategy(Strategy):
    def generate_signals(self,
                        data: pd.DataFrame,
                        features: Optional[pd.DataFrame] = None) -> List[Signal]:
        signals: List[Signal] = []
        # Strategy logic here
        return signals
""",
                language="python"
            ),
            antipattern=CodeExample(
                title="No Type Hints",
                description="Avoid untyped function signatures",
                code="""
class MyStrategy:
    def generate_signals(self, data, features=None):
        signals = []
        # Hard to understand what types are expected
        return signals
""",
                language="python"
            )
        )

        strategy_practices.add_practice(
            title="Implement Proper Error Handling",
            description="Always handle potential errors gracefully in trading strategies",
            rationale="Trading systems must be robust and handle market data issues, API failures, and edge cases",
            importance="critical",
            example=CodeExample(
                title="Robust Error Handling",
                description="Proper error handling in strategy execution",
                code="""
import logging
from typing import List
from qframe.core.interfaces import Signal

def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
    try:
        if data.empty:
            logging.warning("Empty data received, returning no signals")
            return []

        # Strategy logic with validation
        if len(data) < self.min_data_points:
            logging.info("Insufficient data for signal generation")
            return []

        signals = self._compute_signals(data)
        return signals

    except Exception as e:
        logging.error(f"Signal generation failed: {e}")
        return []  # Fail gracefully
""",
                language="python"
            )
        )

        self.add_best_practices(strategy_practices)

        # Code Examples
        data_example = CodeExample(
            title="Fetching Market Data",
            description="How to fetch and use market data in QFrame",
            code="""
from qframe.data.providers.binance_provider import BinanceProvider
from qframe.core.interfaces import TimeFrame
import asyncio

async def fetch_data_example():
    provider = BinanceProvider()

    # Fetch hourly BTCUSDT data
    data = await provider.fetch_ohlcv(
        symbol="BTCUSDT",
        timeframe=TimeFrame.HOUR_1,
        limit=100
    )

    print(f"Fetched {len(data)} candles")
    print(data.head())

    return data

# Run the example
asyncio.run(fetch_data_example())
""",
            language="python",
            tags=["data", "api", "binance"],
            explanation="This example shows how to use the Binance data provider to fetch OHLCV data asynchronously."
        )

        self.add_example(data_example)

    def add_tutorial(self, tutorial: InteractiveTutorial) -> str:
        """Add a tutorial to the portal."""
        tutorial_id = tutorial.title.lower().replace(' ', '_')
        self.tutorials[tutorial_id] = tutorial
        self._update_content_index(tutorial_id, 'tutorial', tutorial.to_dict())
        return tutorial_id

    def add_example(self, example: CodeExample) -> str:
        """Add a code example to the portal."""
        example_id = example.title.lower().replace(' ', '_')
        self.examples[example_id] = example
        self._update_content_index(example_id, 'example', example.to_dict())
        return example_id

    def add_best_practices(self, practices: BestPractices) -> str:
        """Add best practices to the portal."""
        practices_id = practices.title.lower().replace(' ', '_')
        self.best_practices[practices_id] = practices
        self._update_content_index(practices_id, 'best_practices', practices.to_dict())
        return practices_id

    def get_tutorial(self, tutorial_id: str) -> Optional[InteractiveTutorial]:
        """Get a tutorial by ID."""
        return self.tutorials.get(tutorial_id)

    def get_example(self, example_id: str) -> Optional[CodeExample]:
        """Get a code example by ID."""
        return self.examples.get(example_id)

    def get_best_practices(self, practices_id: str) -> Optional[BestPractices]:
        """Get best practices by ID."""
        return self.best_practices.get(practices_id)

    def search_content(self, query: str, content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search through all documentation content."""
        results = []
        query_lower = query.lower()

        for content_id, content_info in self.content_index.items():
            if content_type and content_info['type'] != content_type:
                continue

            # Search in title, description, and tags
            searchable_text = (
                content_info['content']['title'].lower() +
                ' ' + content_info['content'].get('description', '').lower() +
                ' ' + ' '.join(content_info['content'].get('tags', [])).lower()
            )

            if query_lower in searchable_text:
                results.append({
                    'id': content_id,
                    'type': content_info['type'],
                    'title': content_info['content']['title'],
                    'description': content_info['content'].get('description', ''),
                    'relevance': self._calculate_relevance(query_lower, searchable_text)
                })

        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results

    def generate_static_site(self, output_dir: Path) -> None:
        """Generate static HTML documentation site."""
        output_dir.mkdir(exist_ok=True)

        # Create index page
        self._generate_index_page(output_dir)

        # Generate tutorial pages
        tutorials_dir = output_dir / 'tutorials'
        tutorials_dir.mkdir(exist_ok=True)
        for tutorial_id, tutorial in self.tutorials.items():
            self._generate_tutorial_page(tutorial, tutorials_dir / f"{tutorial_id}.html")

        # Generate examples pages
        examples_dir = output_dir / 'examples'
        examples_dir.mkdir(exist_ok=True)
        for example_id, example in self.examples.items():
            self._generate_example_page(example, examples_dir / f"{example_id}.html")

        # Generate best practices pages
        practices_dir = output_dir / 'best-practices'
        practices_dir.mkdir(exist_ok=True)
        for practices_id, practices in self.best_practices.items():
            self._generate_practices_page(practices, practices_dir / f"{practices_id}.html")

        # Copy static assets
        self._copy_static_assets(output_dir)

    def _update_content_index(self, content_id: str, content_type: str, content_data: Dict[str, Any]) -> None:
        """Update the content search index."""
        self.content_index[content_id] = {
            'type': content_type,
            'content': content_data,
            'updated_at': datetime.now().isoformat()
        }

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate search relevance score."""
        # Simple relevance calculation
        words = query.split()
        score = 0.0

        for word in words:
            if word in text:
                score += 1.0 / len(words)

        return score

    def _generate_index_page(self, output_dir: Path) -> None:
        """Generate the main index page."""
        index_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QFrame Documentation Portal</title>
    <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>QFrame Documentation Portal</h1>
            <p>Comprehensive documentation for the quantitative trading framework</p>
        </header>

        <section class="content-overview">
            <div class="content-type">
                <h2>Tutorials</h2>
                <p>{len(self.tutorials)} interactive tutorials</p>
                <a href="tutorials/">Browse Tutorials →</a>
            </div>

            <div class="content-type">
                <h2>Code Examples</h2>
                <p>{len(self.examples)} code examples</p>
                <a href="examples/">Browse Examples →</a>
            </div>

            <div class="content-type">
                <h2>Best Practices</h2>
                <p>{len(self.best_practices)} best practice guides</p>
                <a href="best-practices/">Browse Best Practices →</a>
            </div>
        </section>

        <section class="quick-start">
            <h2>Quick Start</h2>
            <p>New to QFrame? Start with our <a href="tutorials/qframe_quick_start_guide.html">Quick Start Guide</a></p>
        </section>
    </div>
    <script src="assets/scripts.js"></script>
</body>
</html>
        """

        with open(output_dir / 'index.html', 'w') as f:
            f.write(index_content)

    def _generate_tutorial_page(self, tutorial: InteractiveTutorial, output_file: Path) -> None:
        """Generate HTML page for a tutorial."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{tutorial.title} - QFrame Docs</title>
    <link rel="stylesheet" href="../assets/styles.css">
    <link rel="stylesheet" href="../assets/highlight.css">
</head>
<body>
    <div class="container">
        <nav class="breadcrumb">
            <a href="../index.html">Home</a> →
            <a href="index.html">Tutorials</a> →
            {tutorial.title}
        </nav>

        {tutorial.to_html()}

        <footer>
            <p>Last updated: {tutorial.updated_at.strftime('%Y-%m-%d')}</p>
        </footer>
    </div>
    <script src="../assets/scripts.js"></script>
</body>
</html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

    def _generate_example_page(self, example: CodeExample, output_file: Path) -> None:
        """Generate HTML page for a code example."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{example.title} - QFrame Docs</title>
    <link rel="stylesheet" href="../assets/styles.css">
    <link rel="stylesheet" href="../assets/highlight.css">
</head>
<body>
    <div class="container">
        <nav class="breadcrumb">
            <a href="../index.html">Home</a> →
            <a href="index.html">Examples</a> →
            {example.title}
        </nav>

        {example.to_html()}

        <footer>
            <p>Tags: {', '.join(example.tags)}</p>
            <p>Last updated: {example.updated_at.strftime('%Y-%m-%d')}</p>
        </footer>
    </div>
    <script src="../assets/scripts.js"></script>
</body>
</html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

    def _generate_practices_page(self, practices: BestPractices, output_file: Path) -> None:
        """Generate HTML page for best practices."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{practices.title} - QFrame Docs</title>
    <link rel="stylesheet" href="../assets/styles.css">
    <link rel="stylesheet" href="../assets/highlight.css">
</head>
<body>
    <div class="container">
        <nav class="breadcrumb">
            <a href="../index.html">Home</a> →
            <a href="index.html">Best Practices</a> →
            {practices.title}
        </nav>

        {practices.to_html()}

        <footer>
            <p>Category: {practices.category}</p>
            <p>Last updated: {practices.updated_at.strftime('%Y-%m-%d')}</p>
        </footer>
    </div>
    <script src="../assets/scripts.js"></script>
</body>
</html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

    def _copy_static_assets(self, output_dir: Path) -> None:
        """Copy static CSS and JS assets."""
        assets_dir = output_dir / 'assets'
        assets_dir.mkdir(exist_ok=True)

        # Basic CSS
        css_content = """
/* QFrame Documentation Styles */
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.breadcrumb { margin-bottom: 20px; font-size: 14px; }
.breadcrumb a { color: #007acc; text-decoration: none; }
.code-example { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
.example-header { background: #f5f5f5; padding: 10px; border-bottom: 1px solid #ddd; }
.difficulty-badge { padding: 2px 8px; border-radius: 3px; font-size: 12px; }
.difficulty-badge.beginner { background: #d4edda; color: #155724; }
.difficulty-badge.intermediate { background: #fff3cd; color: #856404; }
.difficulty-badge.advanced { background: #f8d7da; color: #721c24; }
.highlight { padding: 15px; overflow-x: auto; }
.run-code-btn, .copy-code-btn { margin: 5px; padding: 5px 10px; }
        """

        with open(assets_dir / 'styles.css', 'w') as f:
            f.write(css_content)

        # Basic JavaScript
        js_content = """
// QFrame Documentation Scripts
function runExample(button) {
    const codeBlock = button.closest('.code-example').querySelector('code');
    const outputDiv = button.closest('.code-example').querySelector('.example-output');

    // This would send code to backend for execution
    outputDiv.style.display = 'block';
    outputDiv.querySelector('.output-content').textContent = 'Code execution would happen here...';
}

function copyCode(button) {
    const codeBlock = button.closest('.code-example').querySelector('code');
    navigator.clipboard.writeText(codeBlock.textContent);

    const originalText = button.textContent;
    button.textContent = 'Copied!';
    setTimeout(() => button.textContent = originalText, 2000);
}

function completeStep(button) {
    const step = button.closest('.tutorial-step');
    step.classList.add('completed');
    button.textContent = 'Completed ✓';
    button.disabled = true;
}

function nextStep(button) {
    const currentStep = button.closest('.tutorial-step');
    const nextStep = currentStep.nextElementSibling;

    if (nextStep && nextStep.classList.contains('tutorial-step')) {
        nextStep.scrollIntoView({ behavior: 'smooth' });
    }
}
        """

        with open(assets_dir / 'scripts.js', 'w') as f:
            f.write(js_content)

        # Syntax highlighting CSS (simplified)
        highlight_css = """
.highlight { background: #f8f8f8; }
.highlight .k { color: #008000; font-weight: bold; } /* Keyword */
.highlight .s { color: #ba2121; } /* String */
.highlight .c { color: #408080; font-style: italic; } /* Comment */
.highlight .n { color: #000000; } /* Name */
        """

        with open(assets_dir / 'highlight.css', 'w') as f:
            f.write(highlight_css)