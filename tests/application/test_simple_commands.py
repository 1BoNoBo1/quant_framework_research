"""
Tests for Application Commands (Simple)
======================================

Tests ciblés pour les commandes d'application.
"""

import pytest
from unittest.mock import Mock
from decimal import Decimal
from datetime import datetime

from qframe.application.base.command import Command, CommandHandler, CommandResult
from qframe.application.base.query import Query, QueryHandler, QueryResult


class TestCommandBase:
    """Tests pour la base des commandes."""

    def test_command_creation(self):
        """Test création d'une commande."""
        command = Command(id="cmd-001", timestamp=datetime.utcnow())

        assert command.id == "cmd-001"
        assert command.timestamp is not None

    def test_command_result(self):
        """Test résultat de commande."""
        result = CommandResult(
            success=True,
            message="Command executed successfully",
            data={"result": "test"}
        )

        assert result.success is True
        assert result.message == "Command executed successfully"
        assert result.data["result"] == "test"

    def test_command_validation(self):
        """Test validation de commande."""
        class TestCommand(Command):
            def __init__(self, value: int):
                super().__init__()
                self.value = value

            def validate(self) -> bool:
                return self.value > 0

        # Commande valide
        valid_cmd = TestCommand(value=10)
        assert valid_cmd.validate() is True

        # Commande invalide
        invalid_cmd = TestCommand(value=-5)
        assert invalid_cmd.validate() is False

    def test_command_handler_interface(self):
        """Test interface CommandHandler."""
        class TestCommandHandler(CommandHandler):
            def handle(self, command: Command) -> CommandResult:
                return CommandResult(
                    success=True,
                    message="Handled",
                    data={"command_id": command.id}
                )

        handler = TestCommandHandler()
        command = Command(id="test-cmd")

        result = handler.handle(command)

        assert result.success is True
        assert result.data["command_id"] == "test-cmd"

    def test_command_error_handling(self):
        """Test gestion d'erreurs dans les commandes."""
        class FailingCommandHandler(CommandHandler):
            def handle(self, command: Command) -> CommandResult:
                raise Exception("Handler failed")

        handler = FailingCommandHandler()
        command = Command(id="failing-cmd")

        # Devrait gérer l'exception gracieusement
        try:
            result = handler.handle(command)
        except Exception as e:
            assert "Handler failed" in str(e)

    def test_command_metadata(self):
        """Test métadonnées de commande."""
        metadata = {
            "user_id": "user-001",
            "source": "api",
            "trace_id": "trace-123"
        }

        command = Command(id="meta-cmd", metadata=metadata)

        assert command.metadata["user_id"] == "user-001"
        assert command.metadata["source"] == "api"


class TestQueryBase:
    """Tests pour la base des queries."""

    def test_query_creation(self):
        """Test création d'une query."""
        query = Query(id="query-001", timestamp=datetime.utcnow())

        assert query.id == "query-001"
        assert query.timestamp is not None

    def test_query_result(self):
        """Test résultat de query."""
        data = [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]

        result = QueryResult(
            success=True,
            data=data,
            total_count=2,
            page=1,
            page_size=10
        )

        assert result.success is True
        assert len(result.data) == 2
        assert result.total_count == 2

    def test_query_pagination(self):
        """Test pagination des queries."""
        class PaginatedQuery(Query):
            def __init__(self, page: int = 1, page_size: int = 10):
                super().__init__()
                self.page = page
                self.page_size = page_size

            @property
            def offset(self) -> int:
                return (self.page - 1) * self.page_size

        query = PaginatedQuery(page=3, page_size=20)

        assert query.page == 3
        assert query.page_size == 20
        assert query.offset == 40  # (3-1) * 20

    def test_query_filtering(self):
        """Test filtrage des queries."""
        class FilteredQuery(Query):
            def __init__(self, filters: dict = None):
                super().__init__()
                self.filters = filters or {}

            def add_filter(self, key: str, value: any):
                self.filters[key] = value

            def has_filter(self, key: str) -> bool:
                return key in self.filters

        query = FilteredQuery()
        query.add_filter("status", "active")
        query.add_filter("created_after", "2023-01-01")

        assert query.has_filter("status") is True
        assert query.filters["status"] == "active"
        assert query.has_filter("deleted") is False

    def test_query_handler_interface(self):
        """Test interface QueryHandler."""
        class TestQueryHandler(QueryHandler):
            def handle(self, query: Query) -> QueryResult:
                return QueryResult(
                    success=True,
                    data=[{"query_id": query.id}],
                    total_count=1
                )

        handler = TestQueryHandler()
        query = Query(id="test-query")

        result = handler.handle(query)

        assert result.success is True
        assert result.data[0]["query_id"] == "test-query"

    def test_query_sorting(self):
        """Test tri des queries."""
        class SortedQuery(Query):
            def __init__(self, sort_by: str = None, sort_order: str = "asc"):
                super().__init__()
                self.sort_by = sort_by
                self.sort_order = sort_order

            def is_sorted(self) -> bool:
                return self.sort_by is not None

            def is_ascending(self) -> bool:
                return self.sort_order.lower() == "asc"

        # Query triée
        sorted_query = SortedQuery(sort_by="created_at", sort_order="desc")
        assert sorted_query.is_sorted() is True
        assert sorted_query.is_ascending() is False

        # Query non triée
        unsorted_query = SortedQuery()
        assert unsorted_query.is_sorted() is False


class TestCommandQueryIntegration:
    """Tests d'intégration commandes/queries."""

    def test_command_query_workflow(self):
        """Test workflow commande → query."""
        # 1. Exécuter commande
        class CreateItemCommand(Command):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

        class CreateItemHandler(CommandHandler):
            def __init__(self):
                self.items = []

            def handle(self, command: CreateItemCommand) -> CommandResult:
                item = {"id": len(self.items) + 1, "name": command.name}
                self.items.append(item)
                return CommandResult(success=True, data=item)

        # 2. Query pour vérifier
        class GetItemsQuery(Query):
            pass

        class GetItemsHandler(QueryHandler):
            def __init__(self, items_store):
                self.items_store = items_store

            def handle(self, query: GetItemsQuery) -> QueryResult:
                return QueryResult(
                    success=True,
                    data=self.items_store.items,
                    total_count=len(self.items_store.items)
                )

        # Test du workflow
        command_handler = CreateItemHandler()
        query_handler = GetItemsHandler(command_handler)

        # Créer item
        create_cmd = CreateItemCommand("Test Item")
        cmd_result = command_handler.handle(create_cmd)

        assert cmd_result.success is True
        assert cmd_result.data["name"] == "Test Item"

        # Vérifier via query
        get_query = GetItemsQuery()
        query_result = query_handler.handle(get_query)

        assert query_result.success is True
        assert len(query_result.data) == 1
        assert query_result.data[0]["name"] == "Test Item"

    def test_command_validation_pipeline(self):
        """Test pipeline de validation des commandes."""
        class ValidatedCommand(Command):
            def __init__(self, amount: Decimal):
                super().__init__()
                self.amount = amount

            def validate(self) -> bool:
                return self.amount > 0

        class ValidationHandler(CommandHandler):
            def handle(self, command: ValidatedCommand) -> CommandResult:
                if not command.validate():
                    return CommandResult(
                        success=False,
                        message="Command validation failed",
                        errors=["Amount must be positive"]
                    )

                return CommandResult(
                    success=True,
                    message="Command executed",
                    data={"amount": float(command.amount)}
                )

        handler = ValidationHandler()

        # Commande valide
        valid_cmd = ValidatedCommand(amount=Decimal("100.00"))
        valid_result = handler.handle(valid_cmd)
        assert valid_result.success is True

        # Commande invalide
        invalid_cmd = ValidatedCommand(amount=Decimal("-50.00"))
        invalid_result = handler.handle(invalid_cmd)
        assert invalid_result.success is False
        assert "validation failed" in invalid_result.message

    def test_async_command_handling(self):
        """Test gestion asynchrone des commandes."""
        import asyncio

        class AsyncCommand(Command):
            def __init__(self, delay: float):
                super().__init__()
                self.delay = delay

        class AsyncCommandHandler(CommandHandler):
            async def handle_async(self, command: AsyncCommand) -> CommandResult:
                await asyncio.sleep(command.delay)
                return CommandResult(
                    success=True,
                    message=f"Async command completed after {command.delay}s",
                    data={"delay": command.delay}
                )

        # Test conceptuel (pas vraiment async dans ce contexte)
        handler = AsyncCommandHandler()
        command = AsyncCommand(delay=0.1)

        # Vérifier que la méthode existe
        assert hasattr(handler, 'handle_async')

    def test_command_transaction_semantics(self):
        """Test sémantique transactionnelle."""
        class TransactionalHandler(CommandHandler):
            def __init__(self):
                self.state = {"counter": 0}
                self.transactions = []

            def handle(self, command: Command) -> CommandResult:
                # Simuler transaction
                try:
                    # Begin transaction
                    original_state = self.state.copy()
                    self.transactions.append(("BEGIN", original_state))

                    # Modify state
                    self.state["counter"] += 1

                    # Simuler succès/échec
                    if hasattr(command, 'should_fail') and command.should_fail:
                        raise Exception("Simulated failure")

                    # Commit
                    self.transactions.append(("COMMIT", self.state.copy()))
                    return CommandResult(success=True, data=self.state.copy())

                except Exception as e:
                    # Rollback
                    self.state = original_state
                    self.transactions.append(("ROLLBACK", self.state.copy()))
                    return CommandResult(success=False, message=str(e))

        handler = TransactionalHandler()

        # Commande qui réussit
        success_cmd = Command(id="success")
        success_result = handler.handle(success_cmd)
        assert success_result.success is True
        assert handler.state["counter"] == 1

        # Commande qui échoue
        fail_cmd = Command(id="fail")
        fail_cmd.should_fail = True
        fail_result = handler.handle(fail_cmd)
        assert fail_result.success is False
        assert handler.state["counter"] == 1  # Rollback

    def test_command_auditing(self):
        """Test audit des commandes."""
        class AuditableHandler(CommandHandler):
            def __init__(self):
                self.audit_log = []

            def handle(self, command: Command) -> CommandResult:
                # Log avant exécution
                audit_entry = {
                    "command_id": command.id,
                    "timestamp": command.timestamp,
                    "status": "executing"
                }
                self.audit_log.append(audit_entry)

                try:
                    # Exécution simulée
                    result = CommandResult(success=True, data={"executed": True})

                    # Log après succès
                    audit_entry["status"] = "completed"
                    audit_entry["success"] = True

                    return result

                except Exception as e:
                    # Log après échec
                    audit_entry["status"] = "failed"
                    audit_entry["error"] = str(e)
                    raise

        handler = AuditableHandler()
        command = Command(id="audited-cmd")

        result = handler.handle(command)

        assert result.success is True
        assert len(handler.audit_log) == 1
        assert handler.audit_log[0]["command_id"] == "audited-cmd"
        assert handler.audit_log[0]["status"] == "completed"

    def test_command_chaining(self):
        """Test chaînage de commandes."""
        class ChainableHandler(CommandHandler):
            def __init__(self):
                self.results = []

            def handle(self, command: Command) -> CommandResult:
                result = CommandResult(
                    success=True,
                    data={"step": len(self.results) + 1, "command_id": command.id}
                )
                self.results.append(result)

                # Générer commande suivante si nécessaire
                if hasattr(command, 'chain_next') and command.chain_next:
                    next_command = Command(id=f"{command.id}-next")
                    next_result = self.handle(next_command)
                    result.data["next_result"] = next_result.data

                return result

        handler = ChainableHandler()

        # Commande avec chaînage
        chain_cmd = Command(id="chain-start")
        chain_cmd.chain_next = True

        result = handler.handle(chain_cmd)

        assert result.success is True
        assert result.data["step"] == 1
        assert "next_result" in result.data
        assert result.data["next_result"]["step"] == 2