"""
Community Platform
==================

Collaborative platform for knowledge sharing, peer learning,
and community-driven research initiatives.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Set
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
import asyncio
import json
import uuid
from pathlib import Path

from qframe.core.container import injectable
from qframe.core.config import FrameworkConfig

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """Rôles utilisateur dans la communauté"""
    NEWCOMER = "newcomer"
    CONTRIBUTOR = "contributor"
    EXPERT = "expert"
    MENTOR = "mentor"
    MODERATOR = "moderator"
    RESEARCHER = "researcher"
    INSTITUTION = "institution"
    VENDOR = "vendor"


class ExpertiseLevel(Enum):
    """Niveaux d'expertise"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    WORLD_CLASS = "world_class"


class ContentType(Enum):
    """Types de contenu communautaire"""
    DISCUSSION = "discussion"
    QUESTION = "question"
    TUTORIAL = "tutorial"
    RESEARCH_PAPER = "research_paper"
    CODE_SNIPPET = "code_snippet"
    CASE_STUDY = "case_study"
    NEWS = "news"
    EVENT = "event"
    JOB_POSTING = "job_posting"


class CollaborationType(Enum):
    """Types de collaboration"""
    RESEARCH_GROUP = "research_group"
    STUDY_GROUP = "study_group"
    COMPETITION = "competition"
    MENTORSHIP = "mentorship"
    PROJECT = "project"
    WORKING_GROUP = "working_group"


@dataclass
class UserProfile:
    """Profil utilisateur communautaire"""
    user_id: str
    username: str
    email: str

    # Informations personnelles
    full_name: Optional[str] = None
    bio: str = ""
    location: Optional[str] = None
    website: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None

    # Statut
    role: UserRole = UserRole.NEWCOMER
    reputation_score: float = 0.0
    expertise_level: ExpertiseLevel = ExpertiseLevel.BEGINNER
    verified: bool = False

    # Expertise et intérêts
    expertise_areas: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    preferred_languages: List[str] = field(default_factory=list)
    skills: Dict[str, float] = field(default_factory=dict)  # skill -> level (0-1)

    # Activité
    join_date: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    posts_count: int = 0
    contributions_count: int = 0
    helped_users: int = 0

    # Récompenses et achievements
    badges: List[str] = field(default_factory=list)
    achievements: List[Dict[str, Any]] = field(default_factory=list)
    total_upvotes: int = 0
    best_answer_count: int = 0

    # Préférences
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    privacy_settings: Dict[str, bool] = field(default_factory=dict)
    mentoring_available: bool = False
    collaboration_interests: List[str] = field(default_factory=list)


@dataclass
class KnowledgeBase:
    """Base de connaissances communautaire"""
    article_id: str
    title: str
    content: str
    author_id: str

    # Catégorisation
    category: str
    tags: List[str]
    difficulty_level: ExpertiseLevel
    content_type: ContentType

    # Qualité
    upvotes: int = 0
    downvotes: int = 0
    views: int = 0
    bookmarks: int = 0

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_reviewed: Optional[datetime] = None

    # Validation
    peer_reviewed: bool = False
    expert_approved: bool = False
    accuracy_score: float = 0.0
    helpfulness_score: float = 0.0

    # Contenu additionnel
    code_examples: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    related_articles: List[str] = field(default_factory=list)


@dataclass
class PeerReview:
    """Review par un pair"""
    review_id: str
    content_id: str
    reviewer_id: str

    # Évaluation
    accuracy_rating: float  # 0-10
    clarity_rating: float
    usefulness_rating: float
    completeness_rating: float

    # Commentaires
    summary: str
    strengths: List[str]
    improvements: List[str]
    detailed_feedback: str

    # Recommandation
    recommendation: str  # "approve", "revise", "reject"
    confidence_level: float  # 0-1

    # Métadonnées
    review_date: datetime = field(default_factory=datetime.utcnow)
    time_spent_minutes: int = 0
    reviewer_expertise_match: float = 0.0


@dataclass
class CollaborationSpace:
    """Espace de collaboration"""
    space_id: str
    name: str
    description: str
    type: CollaborationType

    # Membres
    creator_id: str
    members: List[str] = field(default_factory=list)
    moderators: List[str] = field(default_factory=list)
    max_members: int = 50

    # État
    status: str = "active"  # active, completed, suspended
    privacy: str = "public"  # public, private, invite_only

    # Contenu
    discussions: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)

    # Métadonnées
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    # Activité
    last_activity: datetime = field(default_factory=datetime.utcnow)
    total_messages: int = 0
    active_members: int = 0


@injectable
class CommunityPlatform:
    """
    Plateforme communautaire pour QFrame.

    Fonctionnalités:
    - Gestion de profils utilisateur et réputation
    - Forums de discussion et Q&A
    - Base de connaissances collaborative
    - Système de mentoring et peer review
    - Espaces de collaboration pour projets
    - Événements et compétitions
    - Système de badges et achievements
    - Analyse de sentiment et engagement
    """

    def __init__(self, config: FrameworkConfig):
        self.config = config

        # Storage
        self.users: Dict[str, UserProfile] = {}
        self.knowledge_base: Dict[str, KnowledgeBase] = {}
        self.collaboration_spaces: Dict[str, CollaborationSpace] = {}
        self.peer_reviews: Dict[str, PeerReview] = {}

        # Community metrics
        self.engagement_metrics = {
            "daily_active_users": 0,
            "monthly_active_users": 0,
            "posts_per_day": 0,
            "questions_answered": 0,
            "collaboration_success_rate": 0.0
        }

        # Événements
        self.events: List[Dict[str, Any]] = []
        self.competitions: List[Dict[str, Any]] = []

        logger.info("Community Platform initialized")

    async def create_user_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """Crée un profil utilisateur"""

        user_id = str(uuid.uuid4())

        profile = UserProfile(
            user_id=user_id,
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data.get("full_name"),
            bio=user_data.get("bio", ""),
            location=user_data.get("location"),
            expertise_areas=user_data.get("expertise_areas", []),
            interests=user_data.get("interests", []),
            preferred_languages=user_data.get("preferred_languages", ["English"]),
            expertise_level=user_data.get("expertise_level", ExpertiseLevel.BEGINNER)
        )

        # Configuration initiale
        profile.notification_preferences = {
            "mentions": True,
            "replies": True,
            "new_posts": False,
            "events": True,
            "collaborations": True
        }

        profile.privacy_settings = {
            "profile_public": True,
            "email_visible": False,
            "activity_visible": True
        }

        # Stocker
        self.users[user_id] = profile

        # Assigner badges de bienvenue
        await self._assign_welcome_badges(profile)

        logger.info(f"User profile created: {user_data['username']}")
        return profile

    async def create_knowledge_article(
        self,
        author_id: str,
        article_data: Dict[str, Any]
    ) -> KnowledgeBase:
        """Crée un article dans la base de connaissances"""

        if author_id not in self.users:
            raise ValueError("Author not found")

        article_id = str(uuid.uuid4())

        article = KnowledgeBase(
            article_id=article_id,
            title=article_data["title"],
            content=article_data["content"],
            author_id=author_id,
            category=article_data["category"],
            tags=article_data.get("tags", []),
            difficulty_level=ExpertiseLevel(article_data.get("difficulty", "intermediate")),
            content_type=ContentType(article_data.get("content_type", "discussion")),
            code_examples=article_data.get("code_examples", []),
            references=article_data.get("references", [])
        )

        # Stocker
        self.knowledge_base[article_id] = article

        # Mettre à jour profile auteur
        author = self.users[author_id]
        author.posts_count += 1
        author.contributions_count += 1

        # Vérifier si éligible pour peer review
        if article.content_type in [ContentType.TUTORIAL, ContentType.RESEARCH_PAPER]:
            await self._request_peer_review(article)

        logger.info(f"Knowledge article created: {article.title}")
        return article

    async def create_collaboration_space(
        self,
        creator_id: str,
        space_data: Dict[str, Any]
    ) -> CollaborationSpace:
        """Crée un espace de collaboration"""

        if creator_id not in self.users:
            raise ValueError("Creator not found")

        space_id = str(uuid.uuid4())

        space = CollaborationSpace(
            space_id=space_id,
            name=space_data["name"],
            description=space_data["description"],
            type=CollaborationType(space_data["type"]),
            creator_id=creator_id,
            privacy=space_data.get("privacy", "public"),
            max_members=space_data.get("max_members", 50),
            tags=space_data.get("tags", []),
            deadline=space_data.get("deadline")
        )

        # Ajouter créateur comme membre et modérateur
        space.members.append(creator_id)
        space.moderators.append(creator_id)

        # Stocker
        self.collaboration_spaces[space_id] = space

        # Proposer membres potentiels
        await self._suggest_collaboration_members(space)

        logger.info(f"Collaboration space created: {space.name}")
        return space

    async def submit_peer_review(
        self,
        reviewer_id: str,
        content_id: str,
        review_data: Dict[str, Any]
    ) -> PeerReview:
        """Soumet une peer review"""

        if reviewer_id not in self.users:
            raise ValueError("Reviewer not found")

        if content_id not in self.knowledge_base:
            raise ValueError("Content not found")

        review_id = str(uuid.uuid4())

        review = PeerReview(
            review_id=review_id,
            content_id=content_id,
            reviewer_id=reviewer_id,
            accuracy_rating=review_data["accuracy_rating"],
            clarity_rating=review_data["clarity_rating"],
            usefulness_rating=review_data["usefulness_rating"],
            completeness_rating=review_data["completeness_rating"],
            summary=review_data["summary"],
            strengths=review_data.get("strengths", []),
            improvements=review_data.get("improvements", []),
            detailed_feedback=review_data.get("detailed_feedback", ""),
            recommendation=review_data["recommendation"],
            confidence_level=review_data.get("confidence_level", 0.8),
            time_spent_minutes=review_data.get("time_spent_minutes", 30)
        )

        # Calculer expertise match
        reviewer = self.users[reviewer_id]
        article = self.knowledge_base[content_id]
        review.reviewer_expertise_match = self._calculate_expertise_match(
            reviewer.expertise_areas, article.tags
        )

        # Stocker
        self.peer_reviews[review_id] = review

        # Mettre à jour article
        await self._process_peer_review(review)

        # Récompenser reviewer
        await self._reward_peer_reviewer(reviewer_id, review)

        logger.info(f"Peer review submitted: {review_id}")
        return review

    async def join_collaboration(self, user_id: str, space_id: str) -> bool:
        """Rejoint un espace de collaboration"""

        if user_id not in self.users:
            raise ValueError("User not found")

        if space_id not in self.collaboration_spaces:
            raise ValueError("Collaboration space not found")

        space = self.collaboration_spaces[space_id]

        # Vérifications
        if user_id in space.members:
            return False  # Déjà membre

        if len(space.members) >= space.max_members:
            return False  # Espace plein

        if space.privacy == "invite_only":
            return False  # Invitation requise

        # Ajouter membre
        space.members.append(user_id)
        space.active_members += 1

        # Mettre à jour activité
        space.last_activity = datetime.utcnow()

        # Notifier autres membres
        await self._notify_collaboration_members(space, f"New member joined: {self.users[user_id].username}")

        logger.info(f"User {user_id} joined collaboration {space.name}")
        return True

    async def search_knowledge_base(
        self,
        query: str,
        category: Optional[str] = None,
        difficulty: Optional[ExpertiseLevel] = None,
        content_type: Optional[ContentType] = None
    ) -> List[KnowledgeBase]:
        """Recherche dans la base de connaissances"""

        results = []

        for article in self.knowledge_base.values():
            # Filtres
            if category and article.category != category:
                continue

            if difficulty and article.difficulty_level != difficulty:
                continue

            if content_type and article.content_type != content_type:
                continue

            # Recherche textuelle
            score = self._calculate_relevance_score(article, query)
            if score > 0.3:  # Seuil de pertinence
                results.append((article, score))

        # Trier par pertinence et qualité
        results.sort(key=lambda x: (x[1], x[0].upvotes - x[0].downvotes), reverse=True)

        return [article for article, score in results]

    async def get_personalized_recommendations(self, user_id: str) -> Dict[str, List[Any]]:
        """Recommandations personnalisées pour un utilisateur"""

        if user_id not in self.users:
            raise ValueError("User not found")

        user = self.users[user_id]

        recommendations = {
            "articles": await self._recommend_articles(user),
            "collaborations": await self._recommend_collaborations(user),
            "users_to_follow": await self._recommend_users(user),
            "events": await self._recommend_events(user)
        }

        return recommendations

    async def calculate_user_reputation(self, user_id: str) -> float:
        """Calcule la réputation d'un utilisateur"""

        if user_id not in self.users:
            return 0.0

        user = self.users[user_id]

        # Composantes de la réputation
        post_score = user.posts_count * 2
        vote_score = user.total_upvotes * 5
        help_score = user.helped_users * 10
        badge_score = len(user.badges) * 20
        review_score = len([r for r in self.peer_reviews.values() if r.reviewer_id == user_id]) * 15

        # Score basé sur qualité des contributions
        quality_multiplier = 1.0
        if user.best_answer_count > 10:
            quality_multiplier = 1.5
        elif user.best_answer_count > 5:
            quality_multiplier = 1.2

        total_score = (post_score + vote_score + help_score + badge_score + review_score) * quality_multiplier

        # Normaliser (0-100)
        reputation = min(100.0, total_score / 10)

        # Mettre à jour
        user.reputation_score = reputation

        return reputation

    async def organize_event(self, organizer_id: str, event_data: Dict[str, Any]) -> str:
        """Organise un événement communautaire"""

        if organizer_id not in self.users:
            raise ValueError("Organizer not found")

        event_id = str(uuid.uuid4())

        event = {
            "event_id": event_id,
            "title": event_data["title"],
            "description": event_data["description"],
            "type": event_data["type"],  # workshop, webinar, competition, meetup
            "organizer_id": organizer_id,
            "date": event_data["date"],
            "duration_hours": event_data.get("duration_hours", 2),
            "max_participants": event_data.get("max_participants", 100),
            "registration_deadline": event_data.get("registration_deadline"),
            "tags": event_data.get("tags", []),
            "participants": [],
            "waitlist": [],
            "created_at": datetime.utcnow()
        }

        self.events.append(event)

        # Notifier communauté
        await self._notify_community_event(event)

        logger.info(f"Event organized: {event['title']}")
        return event_id

    def get_community_stats(self) -> Dict[str, Any]:
        """Statistiques de la communauté"""

        active_users = [u for u in self.users.values()
                       if (datetime.utcnow() - u.last_active).days <= 30]

        stats = {
            "overview": {
                "total_users": len(self.users),
                "active_users_30d": len(active_users),
                "total_articles": len(self.knowledge_base),
                "collaboration_spaces": len(self.collaboration_spaces),
                "peer_reviews": len(self.peer_reviews)
            },
            "engagement": {
                "posts_last_week": self._count_recent_posts(7),
                "collaborations_active": len([s for s in self.collaboration_spaces.values()
                                            if s.status == "active"]),
                "avg_session_length": 25,  # minutes
                "user_retention_rate": 78.5  # %
            },
            "content_quality": {
                "peer_reviewed_articles": len([a for a in self.knowledge_base.values()
                                             if a.peer_reviewed]),
                "average_article_rating": self._calculate_avg_content_rating(),
                "expert_approved_content": len([a for a in self.knowledge_base.values()
                                              if a.expert_approved])
            },
            "user_levels": {
                role.value: len([u for u in self.users.values() if u.role == role])
                for role in UserRole
            }
        }

        return stats

    # Méthodes privées

    async def _assign_welcome_badges(self, profile: UserProfile):
        """Assigne les badges de bienvenue"""
        profile.badges.extend(["newcomer", "first_profile"])

    async def _request_peer_review(self, article: KnowledgeBase):
        """Demande une peer review pour un article"""
        # Trouver reviewers potentiels
        potential_reviewers = [
            user for user in self.users.values()
            if any(area in article.tags for area in user.expertise_areas)
            and user.role in [UserRole.EXPERT, UserRole.RESEARCHER, UserRole.MENTOR]
        ]

        # Notifier 3 reviewers max
        for reviewer in potential_reviewers[:3]:
            logger.info(f"Requesting peer review from {reviewer.username} for {article.title}")

    def _calculate_expertise_match(self, reviewer_areas: List[str], content_tags: List[str]) -> float:
        """Calcule la correspondance d'expertise"""
        if not reviewer_areas or not content_tags:
            return 0.0

        matches = len(set(reviewer_areas) & set(content_tags))
        return matches / len(content_tags)

    async def _process_peer_review(self, review: PeerReview):
        """Traite une peer review"""
        article = self.knowledge_base[review.content_id]

        # Mettre à jour score de qualité
        avg_rating = (review.accuracy_rating + review.clarity_rating +
                     review.usefulness_rating + review.completeness_rating) / 4

        article.accuracy_score = (article.accuracy_score + avg_rating) / 2

        # Approuver si recommandé
        if review.recommendation == "approve" and review.confidence_level > 0.8:
            article.peer_reviewed = True

    async def _reward_peer_reviewer(self, reviewer_id: str, review: PeerReview):
        """Récompense un peer reviewer"""
        reviewer = self.users[reviewer_id]

        # Points de réputation
        reputation_boost = 5 + (review.time_spent_minutes / 30) * 2
        reviewer.reputation_score += reputation_boost

        # Badge si premier review
        if len([r for r in self.peer_reviews.values() if r.reviewer_id == reviewer_id]) == 1:
            reviewer.badges.append("first_review")

    async def _suggest_collaboration_members(self, space: CollaborationSpace):
        """Suggère des membres pour une collaboration"""
        # Trouver utilisateurs avec expertise pertinente
        suggestions = [
            user for user in self.users.values()
            if any(tag in user.interests for tag in space.tags)
            and user.user_id not in space.members
        ]

        for user in suggestions[:5]:  # Top 5 suggestions
            logger.info(f"Suggesting collaboration {space.name} to {user.username}")

    async def _notify_collaboration_members(self, space: CollaborationSpace, message: str):
        """Notifie les membres d'une collaboration"""
        for member_id in space.members:
            logger.info(f"Notifying {member_id}: {message}")

    def _calculate_relevance_score(self, article: KnowledgeBase, query: str) -> float:
        """Calcule le score de pertinence d'un article"""
        query_lower = query.lower()

        # Score basé sur titre, contenu, tags
        title_score = 1.0 if query_lower in article.title.lower() else 0.0
        content_score = 0.5 if query_lower in article.content.lower() else 0.0
        tag_score = 0.3 if any(query_lower in tag.lower() for tag in article.tags) else 0.0

        # Boost pour qualité
        quality_boost = (article.upvotes - article.downvotes) / 100

        return title_score + content_score + tag_score + quality_boost

    async def _recommend_articles(self, user: UserProfile) -> List[KnowledgeBase]:
        """Recommande des articles à un utilisateur"""
        articles = []

        for article in self.knowledge_base.values():
            # Score basé sur intérêts
            interest_score = len(set(user.interests) & set(article.tags)) / max(len(article.tags), 1)

            # Score basé sur niveau
            level_match = abs(user.expertise_level.value == article.difficulty_level.value)

            # Score de qualité
            quality_score = (article.upvotes - article.downvotes) / 10

            total_score = interest_score + level_match + quality_score

            if total_score > 1.0:
                articles.append((article, total_score))

        articles.sort(key=lambda x: x[1], reverse=True)
        return [article for article, score in articles[:10]]

    async def _recommend_collaborations(self, user: UserProfile) -> List[CollaborationSpace]:
        """Recommande des collaborations"""
        recommendations = []

        for space in self.collaboration_spaces.values():
            if user.user_id in space.members or space.status != "active":
                continue

            # Score basé sur tags
            tag_match = len(set(user.interests) & set(space.tags)) / max(len(space.tags), 1)

            if tag_match > 0.3:
                recommendations.append(space)

        return recommendations[:5]

    async def _recommend_users(self, user: UserProfile) -> List[UserProfile]:
        """Recommande des utilisateurs à suivre"""
        recommendations = []

        for other_user in self.users.values():
            if other_user.user_id == user.user_id:
                continue

            # Score basé sur intérêts communs
            common_interests = len(set(user.interests) & set(other_user.interests))

            # Score de réputation
            reputation_score = other_user.reputation_score / 100

            total_score = common_interests + reputation_score

            if total_score > 1.0:
                recommendations.append((other_user, total_score))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [user_profile for user_profile, score in recommendations[:10]]

    async def _recommend_events(self, user: UserProfile) -> List[Dict[str, Any]]:
        """Recommande des événements"""
        recommendations = []

        for event in self.events:
            if datetime.now() > event["date"]:
                continue

            # Score basé sur tags
            tag_match = len(set(user.interests) & set(event.get("tags", [])))

            if tag_match > 0:
                recommendations.append(event)

        return recommendations[:5]

    async def _notify_community_event(self, event: Dict[str, Any]):
        """Notifie la communauté d'un nouvel événement"""
        logger.info(f"Notifying community about event: {event['title']}")

    def _count_recent_posts(self, days: int) -> int:
        """Compte les posts récents"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        return len([a for a in self.knowledge_base.values() if a.created_at > cutoff])

    def _calculate_avg_content_rating(self) -> float:
        """Calcule la note moyenne du contenu"""
        articles_with_votes = [a for a in self.knowledge_base.values() if a.upvotes + a.downvotes > 0]

        if not articles_with_votes:
            return 0.0

        total_rating = sum(a.upvotes / (a.upvotes + a.downvotes) for a in articles_with_votes)
        return total_rating / len(articles_with_votes)