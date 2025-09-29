"""
Integration tests for Phase 8: Ecosystem & Community Platform
============================================================

Comprehensive test suite validating the complete ecosystem integration
including marketplace, community, plugins, APIs, and documentation.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from qframe.core.config import FrameworkConfig
from qframe.core.container import get_container

# Ecosystem imports
from qframe.ecosystem.marketplace import (
    StrategyMarketplace, StrategyListing, MarketplaceTransaction,
    ListingType, PricingModel, TransactionStatus
)
from qframe.ecosystem.community import (
    CommunityPlatform, UserProfile, CollaborationSpace,
    ExpertiseLevel
)
from qframe.ecosystem.plugins import (
    PluginManager, PluginManifest, BasePlugin, StrategyPlugin,
    PluginType, SecurityLevel, ExtensionPoint
)
from qframe.ecosystem.apis import (
    PublicAPIManager, APIAuthentication, RateLimiter,
    APIKeyType, APIScope, RateLimitRule, RateLimitType
)
from qframe.ecosystem.documentation import (
    DocumentationPortal, InteractiveTutorial, CodeExample,
    BestPractices, DifficultyLevel
)


class TestEcosystemIntegration:
    """Test complete ecosystem integration scenarios."""

    @pytest.fixture
    async def ecosystem_setup(self):
        """Setup complete ecosystem for testing."""
        config = FrameworkConfig()
        # Use a fresh container for testing instead of the global one
        from qframe.core.container import DIContainer
        container = DIContainer()

        # Create temporary directories
        temp_dir = Path(tempfile.mkdtemp())
        plugins_dir = temp_dir / "plugins"
        docs_dir = temp_dir / "docs"

        plugins_dir.mkdir()
        docs_dir.mkdir()

        # Initialize ecosystem components
        marketplace = StrategyMarketplace(config)
        community = CommunityPlatform(config)
        plugin_manager = PluginManager(config)
        api_manager = PublicAPIManager(config)
        docs_portal = DocumentationPortal(config)

        # Setup plugin manager with temp directory
        plugin_manager.plugin_directory = plugins_dir
        docs_portal.base_path = docs_dir

        yield {
            'config': config,
            'marketplace': marketplace,
            'community': community,
            'plugin_manager': plugin_manager,
            'api_manager': api_manager,
            'docs_portal': docs_portal,
            'temp_dir': temp_dir
        }

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_complete_ecosystem_workflow(self, ecosystem_setup):
        """Test a complete ecosystem workflow from user registration to strategy deployment."""
        setup = ecosystem_setup
        marketplace = setup['marketplace']
        community = setup['community']
        plugin_manager = setup['plugin_manager']
        api_manager = setup['api_manager']
        docs_portal = setup['docs_portal']

        # 1. User Registration and Profile Creation
        user_data = {
            'username': 'quant_trader_alice',
            'email': 'alice@example.com',
            'display_name': 'Alice Johnson',
            'bio': 'Quantitative trader with 5 years experience',
            'expertise_areas': ['machine_learning', 'risk_management'],
            'expertise_level': ExpertiseLevel.INTERMEDIATE
        }

        user_profile = await community.create_user_profile(user_data)
        assert user_profile.username == 'quant_trader_alice'
        assert user_profile.expertise_level == ExpertiseLevel.INTERMEDIATE

        # 2. API Key Creation for User
        api_key = api_manager.create_api_key(
            user_id=user_profile.user_id,
            key_type=APIKeyType.PRIVATE,
            scopes=[APIScope.READ_STRATEGIES, APIScope.WRITE_STRATEGIES]
        )
        assert api_key.user_id == user_profile.user_id
        assert APIScope.READ_STRATEGIES in api_key.scopes

        # 3. Documentation Tutorial Completion
        tutorial = InteractiveTutorial(
            title="Advanced Strategy Development",
            description="Learn to develop sophisticated trading strategies",
            difficulty=DifficultyLevel.ADVANCED
        )

        tutorial.add_step(
            title="Setup Development Environment",
            content="Configure your development environment for strategy creation"
        )

        tutorial_id = docs_portal.add_tutorial(tutorial)

        # Mark tutorial step as completed
        tutorial.mark_step_completed(1, user_profile.user_id)
        assert tutorial.user_progress[user_profile.user_id]['completion_rate'] > 0

        # 4. Plugin Development and Installation
        # Create a mock strategy plugin
        plugin_manifest = PluginManifest(
            name="alice_ml_strategy",
            version="1.0.0",
            author="alice@example.com",
            description="Machine learning trading strategy",
            plugin_type=PluginType.STRATEGY,
            entry_point="strategy.AliceMLStrategy",
            security_level=SecurityLevel.TRUSTED,
            permissions=["market_data_read", "portfolio_write"]
        )

        # Create plugin directory structure
        plugin_dir = plugin_manager.plugin_directory / "alice_ml_strategy"
        plugin_dir.mkdir()

        # Save manifest
        plugin_manifest.save(plugin_dir / "manifest.json")

        # Create simple plugin file
        plugin_code = '''
from qframe.ecosystem.plugins import StrategyPlugin
from qframe.core.interfaces import Strategy, Signal
import pandas as pd
from typing import List

class AliceMLStrategy(StrategyPlugin):
    async def initialize(self):
        pass

    async def cleanup(self):
        pass

    def create_strategy(self) -> Strategy:
        return MockStrategy()

class MockStrategy:
    def generate_signals(self, data: pd.DataFrame, features=None) -> List[Signal]:
        return []
'''

        with open(plugin_dir / "strategy.py", 'w') as f:
            f.write(plugin_code)

        # Install and activate plugin
        await plugin_manager.initialize()
        discovered = await plugin_manager.discover_plugins()
        assert len(discovered) > 0

        # 5. Strategy Marketplace Listing
        listing_data = {
            'name': 'Alice ML Strategy - Advanced Machine Learning',
            'description': 'High-performance ML strategy using LSTM and reinforcement learning',
            'strategy_code': plugin_code,
            'category': 'machine_learning',
            'listing_type': ListingType.PREMIUM,
            'pricing_model': PricingModel.ONE_TIME,
            'price': 299.99,
            'tags': ['ml', 'lstm', 'rl', 'advanced'],
            'performance_metrics': {
                'sharpe_ratio': 2.1,
                'max_drawdown': 0.12,
                'annual_return': 0.34
            }
        }

        # Create a mock strategy object for the listing
        class MockStrategy:
            def generate_signals(self, data, features=None):
                return []

        mock_strategy = MockStrategy()
        listing = await marketplace.create_listing(user_profile.user_id, mock_strategy, listing_data)
        assert listing.name == listing_data['name']
        assert listing.owner_id == user_profile.user_id
        assert listing.status.name == 'DRAFT'

        # 6. Peer Review Process
        # Create another user for peer review
        reviewer_data = {
            'username': 'expert_reviewer_bob',
            'email': 'bob@example.com',
            'display_name': 'Bob Expert',
            'expertise_areas': ['machine_learning', 'strategy_review'],
            'expertise_level': ExpertiseLevel.EXPERT
        }

        reviewer_profile = await community.create_user_profile(reviewer_data)

        # Submit peer review
        review_data = {
            'listing_id': listing.listing_id,
            'reviewer_id': reviewer_profile.user_id,
            'overall_rating': 4.5,
            'code_quality_rating': 4.0,
            'performance_rating': 5.0,
            'documentation_rating': 4.0,
            'review_text': 'Excellent strategy with solid ML implementation. Minor documentation improvements needed.',
            'recommendations': ['Add more detailed parameter explanations', 'Include risk analysis section']
        }

        review = await marketplace.submit_review(review_data)
        assert review.overall_rating == 4.5
        assert review.reviewer_id == reviewer_profile.user_id

        # 7. Collaboration Space Creation
        collaboration_data = {
            'name': 'ML Strategy Research Group',
            'description': 'Collaborative space for ML strategy development',
            'creator_id': user_profile.user_id,
            'privacy_level': 'public',
            'collaboration_type': 'research_group'
        }

        collab_space = await community.create_collaboration_space(collaboration_data)
        assert collab_space.name == collaboration_data['name']

        # Add reviewer to collaboration space
        await community.add_member_to_space(collab_space.space_id, reviewer_profile.user_id)

        # 8. Knowledge Base Contribution
        knowledge_data = {
            'title': 'Advanced LSTM Strategies for Crypto Trading',
            'content': 'Comprehensive guide on implementing LSTM networks for cryptocurrency trading...',
            'category': 'machine_learning',
            'author_id': user_profile.user_id,
            'tags': ['lstm', 'crypto', 'neural_networks'],
            'difficulty_level': 'advanced'
        }

        knowledge_article = await community.create_knowledge_article(knowledge_data)
        assert knowledge_article.title == knowledge_data['title']

        # 9. Marketplace Transaction
        # Create buyer user
        buyer_data = {
            'username': 'strategy_buyer_charlie',
            'email': 'charlie@example.com',
            'display_name': 'Charlie Buyer',
            'expertise_level': ExpertiseLevel.BEGINNER
        }

        buyer_profile = await community.create_user_profile(buyer_data)

        # Approve listing first (simulate admin approval)
        await marketplace.update_listing_status(listing.listing_id, 'APPROVED')

        # Execute purchase transaction
        transaction_data = {
            'listing_id': listing.listing_id,
            'buyer_id': buyer_profile.user_id,
            'payment_method': 'credit_card',
            'billing_address': {
                'street': '123 Trading St',
                'city': 'Finance City',
                'country': 'USA'
            }
        }

        transaction = await marketplace.purchase_strategy(transaction_data)
        assert transaction.listing_id == listing.listing_id
        assert transaction.buyer_id == buyer_profile.user_id
        assert transaction.status == TransactionStatus.PENDING

        # 10. API Usage for Strategy Data
        # Test rate limiting
        rate_limiter = api_manager.rate_limiter
        rate_limit_rule = RateLimitRule(
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_limit=10,
            limit_type=RateLimitType.SLIDING_WINDOW
        )

        # Test multiple API calls within rate limit
        for i in range(5):
            allowed, info = rate_limiter.is_allowed(f"test_key_{api_key.key_id}", rate_limit_rule)
            assert allowed, f"Request {i+1} should be allowed"

        # 11. Documentation Generation
        # Add code example for the new strategy
        code_example = CodeExample(
            title="Using Alice ML Strategy",
            description="How to integrate and use the Alice ML Strategy in your trading system",
            code='''
from alice_ml_strategy import AliceMLStrategy
from qframe.core.container import get_container

# Initialize strategy
container = get_container()
strategy = AliceMLStrategy()

# Configure strategy parameters
strategy.configure({
    'learning_rate': 0.001,
    'sequence_length': 60,
    'hidden_units': 128
})

# Generate trading signals
signals = strategy.generate_signals(market_data)
print(f"Generated {len(signals)} trading signals")
''',
            tags=['strategy', 'ml', 'tutorial'],
            difficulty=DifficultyLevel.INTERMEDIATE
        )

        example_id = docs_portal.add_example(code_example)
        assert example_id is not None

        # 12. Verify Complete Integration
        # Check that all components are properly connected

        # User has profile in community
        assert user_profile.user_id in community.user_profiles

        # User has API access
        assert api_key.user_id == user_profile.user_id

        # Strategy is listed in marketplace
        assert listing.listing_id in marketplace.listings

        # Documentation exists
        assert example_id in docs_portal.examples

        # Transaction is recorded
        assert transaction.transaction_id in marketplace.transactions

        # Collaboration space exists
        assert collab_space.space_id in community.collaboration_spaces

        print("✅ Complete ecosystem integration test passed!")

    @pytest.mark.asyncio
    async def test_plugin_security_integration(self, ecosystem_setup):
        """Test plugin security system integration."""
        setup = ecosystem_setup
        plugin_manager = setup['plugin_manager']

        # Test secure plugin installation
        secure_manifest = PluginManifest(
            name="secure_plugin",
            version="1.0.0",
            author="security@example.com",
            description="Secure plugin with limited permissions",
            plugin_type=PluginType.INDICATOR,
            entry_point="plugin.SecurePlugin",
            security_level=SecurityLevel.SANDBOX,
            permissions=["market_data_read"]  # Limited permissions
        )

        # Test security validation
        security_manager = plugin_manager.security_manager

        # Mock plugin for testing
        class MockSecurePlugin(BasePlugin):
            async def initialize(self):
                pass
            async def cleanup(self):
                pass

        mock_plugin = MockSecurePlugin(secure_manifest)

        # Should pass security validation
        is_secure = security_manager.validate_plugin_security(mock_plugin)
        assert is_secure, "Secure plugin should pass validation"

        # Test insecure plugin
        insecure_manifest = PluginManifest(
            name="insecure_plugin",
            version="1.0.0",
            author="hacker@example.com",
            description="Plugin with suspicious permissions",
            plugin_type=PluginType.STRATEGY,
            entry_point="plugin.InsecurePlugin",
            security_level=SecurityLevel.UNRESTRICTED,  # Requires admin approval
            permissions=["system_access", "file_write"]  # Dangerous permissions
        )

        mock_insecure_plugin = MockSecurePlugin(insecure_manifest)

        # Should fail security validation
        is_secure = security_manager.validate_plugin_security(mock_insecure_plugin)
        assert not is_secure, "Insecure plugin should fail validation"

    @pytest.mark.asyncio
    async def test_api_rate_limiting_integration(self, ecosystem_setup):
        """Test API rate limiting across different scenarios."""
        setup = ecosystem_setup
        api_manager = setup['api_manager']

        rate_limiter = api_manager.rate_limiter

        # Test different rate limiting strategies
        test_rules = [
            RateLimitRule(10, 100, 1000, 5, RateLimitType.SLIDING_WINDOW),
            RateLimitRule(15, 150, 1500, 8, RateLimitType.TOKEN_BUCKET),
            RateLimitRule(12, 120, 1200, 6, RateLimitType.FIXED_WINDOW),
            RateLimitRule(20, 200, 2000, 10, RateLimitType.ADAPTIVE)
        ]

        for i, rule in enumerate(test_rules):
            test_key = f"test_key_{i}"

            # Test normal usage within limits
            for j in range(3):  # Well within limit
                allowed, info = rate_limiter.is_allowed(test_key, rule, "127.0.0.1")
                assert allowed, f"Request {j+1} should be allowed for {rule.limit_type}"

            # Test burst capacity
            burst_requests = min(rule.burst_limit, rule.requests_per_minute)
            for j in range(burst_requests):
                allowed, info = rate_limiter.is_allowed(f"{test_key}_burst", rule, "127.0.0.1")
                # At least some should be allowed
                if j < burst_requests // 2:
                    assert allowed, f"Burst request {j+1} should be allowed"

    @pytest.mark.asyncio
    async def test_marketplace_community_integration(self, ecosystem_setup):
        """Test integration between marketplace and community features."""
        setup = ecosystem_setup
        marketplace = setup['marketplace']
        community = setup['community']

        # Create expert user
        expert_data = {
            'username': 'market_expert',
            'email': 'expert@example.com',
            'display_name': 'Market Expert',
            'expertise_areas': ['quantitative_analysis', 'risk_management'],
            'expertise_level': ExpertiseLevel.EXPERT
        }

        expert_profile = await community.create_user_profile(expert_data)

        # Create strategy listing
        listing_data = {
            'title': 'Expert Risk Management Strategy',
            'description': 'Professional risk management system',
            'strategy_code': 'class RiskStrategy: pass',
            'category': 'risk_management',
            'listing_type': ListingType.PREMIUM,
            'pricing_model': PricingModel.SUBSCRIPTION,
            'base_price': 99.99
        }

        listing = await marketplace.create_listing(expert_profile.user_id, listing_data)

        # Test community reputation affecting marketplace
        # High reputation user should have advantages
        assert listing.seller_id == expert_profile.user_id

        # Create collaboration space for strategy discussion
        collab_data = {
            'name': 'Risk Management Discussion',
            'description': 'Discuss risk management strategies',
            'creator_id': expert_profile.user_id,
            'privacy_level': 'public'
        }

        collab_space = await community.create_collaboration_space(collab_data)

        # Link marketplace listing to collaboration space
        listing.discussion_space_id = collab_space.space_id

        # Test cross-platform data consistency
        assert collab_space.creator_id == listing.seller_id

    @pytest.mark.asyncio
    async def test_documentation_integration_workflow(self, ecosystem_setup):
        """Test documentation system integration with other components."""
        setup = ecosystem_setup
        docs_portal = setup['docs_portal']
        community = setup['community']

        # Create technical writer user
        writer_data = {
            'username': 'tech_writer',
            'email': 'writer@example.com',
            'display_name': 'Technical Writer',
            'expertise_areas': ['documentation', 'education'],
            'expertise_level': ExpertiseLevel.INTERMEDIATE
        }

        writer_profile = await community.create_user_profile(writer_data)

        # Create comprehensive tutorial
        tutorial = InteractiveTutorial(
            title="Complete Trading System Setup",
            description="End-to-end guide for setting up a trading system",
            difficulty=DifficultyLevel.INTERMEDIATE,
            estimated_duration=90,
            author=writer_profile.username
        )

        # Add multiple steps
        tutorial.add_step(
            title="Environment Setup",
            content="Set up your development environment",
            code_example=CodeExample(
                title="Installation",
                description="Install QFrame framework",
                code="pip install qframe",
                expected_output="Successfully installed qframe"
            )
        )

        tutorial.add_step(
            title="Strategy Development",
            content="Develop your first strategy",
            quiz_questions=[{
                'question': 'What is the primary interface for strategies?',
                'options': ['Strategy', 'BaseStrategy', 'TradingStrategy'],
                'correct_answer': 0
            }]
        )

        tutorial_id = docs_portal.add_tutorial(tutorial)

        # Test search functionality
        search_results = docs_portal.search_content("trading system", "tutorial")
        assert len(search_results) > 0
        assert any(result['id'] == tutorial_id for result in search_results)

        # Test static site generation
        temp_output = setup['temp_dir'] / "docs_output"
        temp_output.mkdir()

        docs_portal.generate_static_site(temp_output)

        # Verify generated files
        assert (temp_output / "index.html").exists()
        assert (temp_output / "tutorials").exists()
        assert (temp_output / "assets" / "styles.css").exists()

        print("✅ Documentation integration test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])